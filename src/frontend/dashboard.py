"""Read-only AIPulse operations dashboard backed by the API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from .api_client import APIClient, APIError
else:
    from aipulse.frontend.api_client import APIClient, APIError

st.set_page_config(page_title="AIPulse Operations", page_icon="AP", layout="wide")


@st.cache_data(ttl=15, show_spinner=False)
def load_health(base_url: str) -> dict[str, Any]:
    return APIClient(base_url).health()


@st.cache_data(ttl=15, show_spinner=False)
def load_models(base_url: str, token: str) -> dict[str, Any]:
    return APIClient(base_url, token).list_models()


st.title("AIPulse Operations")
st.caption("Live model registry and service health. Source: AIPulse API; cache window: 15 seconds.")

default_url = os.getenv("AIPULSE_API_URL", "http://localhost:8000")
api_url = st.sidebar.text_input("API URL", value=default_url)
access_token = st.sidebar.text_input(
    "Access token",
    type="password",
    help="Use a short-lived token. It is sent only to the configured API URL.",
)
if st.sidebar.button("Refresh data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    health = load_health(api_url)
except (APIError, ValueError) as exc:
    st.error(str(exc))
    st.info("Start the API, verify the URL, then select Refresh data.")
    st.stop()

status = health.get("status", "unknown")
if status == "ok":
    st.success("API is healthy and authentication is configured.")
else:
    st.warning(
        "API is reachable, but authentication is not configured. Protected data is unavailable."
    )

col_service, col_version, col_auth = st.columns(3)
col_service.metric("Service", health.get("service", "unknown"))
col_version.metric("Version", health.get("version", "unknown"))
col_auth.metric(
    "Authentication",
    "Configured" if health.get("authentication_configured") else "Not configured",
)

st.subheader("Registered models")
if not access_token:
    st.info("Enter a viewer, data scientist, ML engineer, or admin token to load model data.")
    st.stop()

try:
    response = load_models(api_url, access_token)
except APIError as exc:
    if exc.status_code in {401, 403}:
        st.error("The token is invalid, expired, or does not have a permitted role.")
    else:
        st.error(str(exc))
    st.stop()

items = response.get("items", [])
if not items:
    st.info("No models have been registered yet. Use the Admin Console to train the first model.")
    st.stop()

rows: list[dict[str, Any]] = []
for item in items:
    metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
    rows.append(
        {
            "Model": item.get("model_name"),
            "Version": item.get("version"),
            "Accuracy": metrics.get("accuracy"),
            "F1": metrics.get("f1_score"),
            "AUC-ROC": metrics.get("auc_roc"),
            "Tags": ", ".join(item.get("tags", [])),
            "Created": item.get("created_at"),
            "Dataset": item.get("training_data_version"),
        }
    )
frame = pd.DataFrame(rows)

model_filter = st.selectbox("Model", ["All", *sorted(frame["Model"].dropna().unique())])
visible = frame if model_filter == "All" else frame[frame["Model"] == model_filter]

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Registered versions", len(visible))
metric_b.metric("Validated versions", int(visible["Tags"].str.contains("validated").sum()))
latest = visible["Created"].dropna().max() if not visible.empty else "—"
metric_c.metric("Latest registration", str(latest))

st.dataframe(
    visible,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Accuracy": st.column_config.NumberColumn(format="%.4f"),
        "F1": st.column_config.NumberColumn(format="%.4f"),
        "AUC-ROC": st.column_config.NumberColumn(format="%.4f"),
    },
)

chart_data = visible.dropna(subset=["Accuracy", "F1"]).set_index("Version")[["Accuracy", "F1"]]
if not chart_data.empty:
    st.subheader("Validation metrics by version")
    st.bar_chart(chart_data, y_label="Score (0-1)", x_label="Model version")
