"""Unified prediction and explanation helpers for the app and notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

try:
    from .io_utils import load_pickle, read_json
    from .sequence_data import rerank_candidates
    from .train_sasrec import SASRecModel
except ImportError:
    from io_utils import load_pickle, read_json
    from sequence_data import rerank_candidates
    from train_sasrec import SASRecModel


def load_xgb_ensemble(artifacts_dir: Path) -> Dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    explainer_metadata = read_json(artifacts_dir / "xgb_explainer_fold.json", default={})
    feature_columns_payload = read_json(artifacts_dir / "feature_columns.json", default={})
    feature_columns = feature_columns_payload.get("model_feature_columns", feature_columns_payload)
    boosters = {}
    for model_path in sorted(artifacts_dir.glob("xgb_fold_*.json")):
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        boosters[int(model_path.stem.split("_")[-1])] = booster
    calibrator_path = artifacts_dir / "calibrator.pkl"
    calibrator = load_pickle(calibrator_path) if calibrator_path.exists() else None
    return {"boosters": boosters, "feature_columns": feature_columns, "explainer_metadata": explainer_metadata, "calibrator": calibrator}


def score_user_success(feature_row: pd.DataFrame, xgb_bundle: Dict[str, Any]) -> Dict[str, float]:
    feature_columns = xgb_bundle["feature_columns"]
    dmatrix = xgb.DMatrix(feature_row[feature_columns], feature_names=feature_columns)
    fold_probs = [float(booster.predict(dmatrix)[0]) for booster in xgb_bundle["boosters"].values()]
    probability = float(np.mean(fold_probs)) if fold_probs else 0.0
    if xgb_bundle["calibrator"] is not None:
        probability = float(xgb_bundle["calibrator"].predict(np.array([[probability]])).ravel()[0])
    return {"success_probability": probability}


def explain_user_prediction(feature_row: pd.DataFrame, xgb_bundle: Dict[str, Any], top_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    feature_columns = xgb_bundle["feature_columns"]
    chosen_fold = int(xgb_bundle["explainer_metadata"].get("chosen_fold", sorted(xgb_bundle["boosters"])[0]))
    booster = xgb_bundle["boosters"][chosen_fold]
    dmatrix = xgb.DMatrix(feature_row[feature_columns], feature_names=feature_columns)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    feature_contribs = pd.Series(contribs[:-1], index=feature_columns).sort_values(ascending=False)
    positive = [{"feature": feature, "contribution": float(value)} for feature, value in feature_contribs.head(top_n).items() if value > 0]
    negative = [{"feature": feature, "contribution": float(value)} for feature, value in feature_contribs.tail(top_n).sort_values().items() if value < 0]
    return {"positive_drivers": positive, "negative_drivers": negative}


def load_sasrec_bundle(artifacts_dir: Path) -> Dict[str, Any]:
    artifacts_dir = Path(artifacts_dir)
    try:
        checkpoint = torch.load(artifacts_dir / "sasrec_model.pt", map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(artifacts_dir / "sasrec_model.pt", map_location="cpu")
    token_to_id = json.loads((artifacts_dir / "token_to_id.json").read_text())
    id_to_token_raw = json.loads((artifacts_dir / "id_to_token.json").read_text())
    id_to_token = {int(key): value for key, value in id_to_token_raw.items()}
    sequence_config = json.loads((artifacts_dir / "sequence_config.json").read_text())
    model = SASRecModel(
        num_items=checkpoint["num_items"],
        max_seq_len=checkpoint["max_seq_len"],
        embedding_dim=checkpoint["embedding_dim"],
        num_blocks=checkpoint["num_blocks"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {"model": model, "token_to_id": token_to_id, "id_to_token": id_to_token, "config": sequence_config}


def predict_next_actions(history_tokens: Iterable[str], sasrec_bundle: Dict[str, Any], transition_lift: pd.DataFrame | None = None, top_k: int = 3) -> List[Dict[str, Any]]:
    history_tokens = list(history_tokens)
    if not history_tokens:
        return []
    token_to_id = sasrec_bundle["token_to_id"]
    id_to_token = sasrec_bundle["id_to_token"]
    model = sasrec_bundle["model"]
    max_seq_len = int(sasrec_bundle["config"]["max_seq_len"])
    encoded = [token_to_id["<PAD>"]] * max(0, max_seq_len - len(history_tokens[-max_seq_len:])) + [
        token_to_id.get(token, token_to_id["<UNK>"]) for token in history_tokens[-max_seq_len:]
    ]
    input_ids = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_candidate_ids = np.argsort(-probabilities)[: max(top_k * 10, 20)]
    candidates = [{"token": id_to_token[int(token_id)], "score": float(probabilities[int(token_id)])} for token_id in top_candidate_ids if int(token_id) in id_to_token]
    return rerank_candidates(candidates, last_token=history_tokens[-1], transition_lift=transition_lift, top_k=top_k)


def closest_successful_archetype(feature_row: pd.DataFrame, archetypes: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
    if archetypes.empty:
        return {"archetype_label": "Unavailable", "distance": float("nan")}
    centroid_columns = [column for column in feature_columns if column in archetypes.columns]
    row_values = feature_row.iloc[0][centroid_columns].astype(float).to_numpy()
    distances = archetypes[centroid_columns].astype(float).apply(lambda centroid: float(np.linalg.norm(row_values - centroid.to_numpy())), axis=1)
    best_idx = int(distances.idxmin())
    best_row = archetypes.loc[best_idx]
    return {
        "archetype_label": best_row.get("archetype_label", f"Archetype {best_idx}"),
        "archetype_summary": best_row.get("archetype_summary", ""),
        "distance": float(distances.loc[best_idx]),
    }
