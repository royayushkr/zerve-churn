"""Sequence-prefix construction, tokenization, and transition-lift helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def build_token_maps(sequence_prefix_table: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = set(sequence_prefix_table["next_token"].astype(str).tolist())
    for prefix_json in sequence_prefix_table["prefix_tokens_json"].astype(str):
        vocab.update(json.loads(prefix_json))
    ordered_vocab = ["<PAD>", "<UNK>"] + sorted(vocab)
    token_to_id = {token: idx for idx, token in enumerate(ordered_vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_history_tokens(history_tokens: Iterable[str], token_to_id: Dict[str, int], max_seq_len: int) -> List[int]:
    history_tokens = list(history_tokens)[-max_seq_len:]
    token_ids = [token_to_id.get(token, token_to_id["<UNK>"]) for token in history_tokens]
    return [token_to_id["<PAD>"]] * (max_seq_len - len(token_ids)) + token_ids


def build_transition_lift(observation_events: pd.DataFrame, successful_user_ids: Iterable[str]) -> pd.DataFrame:
    successful_user_ids = set(successful_user_ids)
    observation_events = observation_events.sort_values(["canonical_user_id", "event_timestamp"]).copy()
    all_pairs = []
    success_pairs = []
    for canonical_user_id, group in observation_events.groupby("canonical_user_id", sort=False):
        tokens = group["event_token_final"].astype(str).tolist()
        pairs = list(zip(tokens[:-1], tokens[1:]))
        all_pairs.extend(pairs)
        if canonical_user_id in successful_user_ids:
            success_pairs.extend(pairs)
    all_df = pd.DataFrame(all_pairs, columns=["from_token", "to_token"])
    success_df = pd.DataFrame(success_pairs, columns=["from_token", "to_token"])
    if all_df.empty:
        return pd.DataFrame(columns=["from_token", "to_token", "all_rate", "success_rate", "lift"])
    all_counts = all_df.value_counts().rename("all_count").reset_index()
    success_counts = success_df.value_counts().rename("success_count").reset_index() if not success_df.empty else pd.DataFrame(columns=["from_token", "to_token", "success_count"])
    merged = all_counts.merge(success_counts, on=["from_token", "to_token"], how="left").fillna({"success_count": 0})
    merged["all_rate"] = merged["all_count"] / max(merged["all_count"].sum(), 1)
    merged["success_rate"] = merged["success_count"] / max(merged["success_count"].sum(), 1)
    merged["lift"] = (merged["success_rate"] + 1e-9) / (merged["all_rate"] + 1e-9)
    return merged.sort_values("lift", ascending=False).reset_index(drop=True)


def is_low_value_token(token: str) -> bool:
    event_name = token.split("|")[0]
    low_value_exact = {
        "sign_in", "sign_up", "link_clicked", "button_clicked", "fullscreen_open",
        "fullscreen_close", "ai_credit_banner_clicked", "ai_credit_banner_shown",
        "clicked_add_credits", "agent_add_credits_button_clicked", "agent_retry_message_button_clicked",
    }
    return event_name in low_value_exact or event_name.startswith("credits_below_")


def rerank_candidates(candidates: List[Dict[str, Any]], last_token: str, transition_lift: pd.DataFrame | None, top_k: int = 3) -> List[Dict[str, Any]]:
    if transition_lift is not None and not transition_lift.empty:
        token_lift = transition_lift.loc[transition_lift["from_token"] == last_token, ["to_token", "lift"]].copy()
        lift_map = token_lift.set_index("to_token")["lift"].to_dict()
    else:
        lift_map = {}
    reranked = []
    for row in candidates:
        if is_low_value_token(row["token"]):
            continue
        reranked.append({**row, "rerank_score": float(row["score"]) * float(lift_map.get(row["token"], 1.0))})
    return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
