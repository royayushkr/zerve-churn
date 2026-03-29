"""Streamlit app for the Zerve hackathon success and recommendation demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SEARCH_ROOTS = [
    APP_DIR,
    APP_DIR.parent,
    Path.cwd(),
]

for root in SEARCH_ROOTS:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    src_dir = root / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

from src.inference import (
    closest_successful_archetype,
    explain_user_prediction,
    load_sasrec_bundle,
    load_xgb_ensemble,
    predict_next_actions,
    score_user_success,
)


ARTIFACT_DIR = Path(os.getenv("ZER_ARTIFACT_DIR", "artifacts"))
DEPLOY_BLOCK_NAME = os.getenv("ZERVE_DEPLOY_BLOCK", "deployment_context")


def try_zerve_variable(block_name: str, variable_name: str) -> Optional[Any]:
    try:
        from zerve import variable

        return variable(block_name, variable_name)
    except Exception:
        return None


def load_manifest() -> Dict[str, Any]:
    manifest_path = ARTIFACT_DIR / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    manifest = try_zerve_variable(DEPLOY_BLOCK_NAME, "deploy_manifest")
    return manifest if manifest is not None else {}


def load_frame_from_disk_or_zerve(variable_name: str, fallback_path: Path) -> pd.DataFrame:
    if fallback_path.exists():
        return pd.read_parquet(fallback_path)
    frame = try_zerve_variable(DEPLOY_BLOCK_NAME, variable_name)
    return frame if frame is not None else pd.DataFrame()


def safe_json_loads(value: Any, default: Any) -> Any:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return json.loads(value)
    except Exception:
        return default


@st.cache_resource(show_spinner=False)
def load_bundles(artifact_dir: str) -> Dict[str, Any]:
    artifact_path = Path(artifact_dir)
    xgb_bundle = load_xgb_ensemble(artifact_path)
    sasrec_bundle = load_sasrec_bundle(artifact_path)
    transition_lift = pd.read_parquet(artifact_path / "transition_lift.parquet") if (artifact_path / "transition_lift.parquet").exists() else pd.DataFrame()
    archetypes = pd.read_parquet(artifact_path / "archetypes.parquet") if (artifact_path / "archetypes.parquet").exists() else pd.DataFrame()
    return {"xgb_bundle": xgb_bundle, "sasrec_bundle": sasrec_bundle, "transition_lift": transition_lift, "archetypes": archetypes}


st.set_page_config(page_title="Zerve Success Analyzer", layout="wide")
st.title("Zerve Success Analyzer")
st.caption("Loads local artifacts first, then notebook variables if running inside Zerve. No training happens inside the app.")

manifest = load_manifest()
artifact_dir = str(ARTIFACT_DIR) if ARTIFACT_DIR.exists() else (try_zerve_variable(DEPLOY_BLOCK_NAME, "deploy_artifact_dir") or str(ARTIFACT_DIR))

demo_users = load_frame_from_disk_or_zerve("deploy_demo_users", Path(artifact_dir) / "demo_users.parquet")
demo_sessions = load_frame_from_disk_or_zerve("deploy_demo_sessions", Path(artifact_dir) / "demo_sessions.parquet")
feature_table = load_frame_from_disk_or_zerve("deploy_feature_table", Path(artifact_dir) / "intermediate" / "user_feature_table.parquet")
events_enriched = load_frame_from_disk_or_zerve("deploy_events_enriched", Path(artifact_dir) / "intermediate" / "events_enriched.parquet")

if demo_users.empty:
    st.warning("Demo users are not available yet. Run the notebook through the deployment block first.")
    st.stop()

bundles = load_bundles(artifact_dir)
xgb_bundle = bundles["xgb_bundle"]
sasrec_bundle = bundles["sasrec_bundle"]
transition_lift = bundles["transition_lift"]
archetypes = bundles["archetypes"]

feature_columns_path = Path(manifest.get("artifact_contract", {}).get("feature_columns_json", ""))
if not feature_columns_path.exists():
    feature_columns_path = Path(artifact_dir) / "feature_columns.json"
feature_columns = json.loads(feature_columns_path.read_text())["model_feature_columns"]

with st.sidebar:
    st.header("Controls")
    selected_user_id = st.selectbox("User selector", options=demo_users["canonical_user_id"].tolist())
    user_sessions = demo_sessions.loc[demo_sessions["canonical_user_id"] == selected_user_id, "session_key"].dropna().astype(str).unique().tolist()
    _ = st.selectbox("Demo session selector", options=user_sessions if user_sessions else ["No session available"])
    model_mode = st.radio("Model mode toggle", options=["Success + Recommendations", "Success Only", "Recommendations Only"], index=0)
    analyze = st.button("Analyze", type="primary")

if not analyze:
    st.info("Select a demo user and click Analyze.")
    st.stop()

user_feature_row = feature_table.loc[feature_table["canonical_user_id"] == selected_user_id].copy()
user_demo_row = demo_users.loc[demo_users["canonical_user_id"] == selected_user_id].copy()
user_timeline = demo_sessions.loc[demo_sessions["canonical_user_id"] == selected_user_id].sort_values("event_timestamp").copy()

if user_feature_row.empty or user_timeline.empty:
    st.warning("This user has insufficient saved history for a full analysis.")
    st.stop()

score_payload = score_user_success(user_feature_row[feature_columns], xgb_bundle)
explain_payload = explain_user_prediction(user_feature_row[feature_columns], xgb_bundle, top_n=5)
history_tokens = user_timeline["event_token_final"].astype(str).tolist() if "event_token_final" in user_timeline.columns else []
recommendations = predict_next_actions(history_tokens, sasrec_bundle, transition_lift=transition_lift, top_k=3) if history_tokens else []
archetype_payload = closest_successful_archetype(user_feature_row[feature_columns], archetypes, feature_columns)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Success Probability", f"{score_payload['success_probability']:.1%}")
with col2:
    st.metric("Percentile", f"{float(user_demo_row['score_percentile'].iloc[0]):.0%}" if "score_percentile" in user_demo_row.columns else "Unavailable")
with col3:
    st.metric("Risk Band", str(user_demo_row["risk_band"].iloc[0]) if "risk_band" in user_demo_row.columns else "Unavailable")

if model_mode in {"Success + Recommendations", "Success Only"}:
    left, right = st.columns(2)
    with left:
        st.subheader("Top Positive Drivers")
        st.dataframe(pd.DataFrame(explain_payload["positive_drivers"])) if explain_payload["positive_drivers"] else st.info("No strong positive drivers were available.")
    with right:
        st.subheader("Top Negative Drivers")
        st.dataframe(pd.DataFrame(explain_payload["negative_drivers"])) if explain_payload["negative_drivers"] else st.info("No strong negative drivers were available.")

st.subheader("Recent Workflow Timeline")
timeline_columns = [column for column in ["event_timestamp", "event", "action_group", "pathname_group", "surface_group", "tool_name_norm", "session_key"] if column in user_timeline.columns]
st.dataframe(user_timeline[timeline_columns].tail(25), use_container_width=True)

if model_mode in {"Success + Recommendations", "Recommendations Only"}:
    st.subheader("Top 3 Recommended Next Actions")
    st.dataframe(pd.DataFrame(recommendations)) if recommendations else st.info("Not enough history to generate recommendations.")

st.subheader("Closest Successful Archetype")
st.write(archetype_payload.get("archetype_label", "Unavailable"))
if archetype_payload.get("archetype_summary"):
    st.caption(archetype_payload["archetype_summary"])

if not user_demo_row.empty and "recommended_actions_json" in user_demo_row.columns:
    precomputed = safe_json_loads(user_demo_row["recommended_actions_json"].iloc[0], [])
    if precomputed:
        st.subheader("Saved Notebook Recommendations")
        st.dataframe(pd.DataFrame(precomputed))
