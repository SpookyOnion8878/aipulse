"""AIPulse model training console backed by authenticated API calls."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from .api_client import APIClient, APIError
else:
    from aipulse.frontend.api_client import APIClient, APIError

st.set_page_config(page_title="AIPulse Admin", page_icon="AP", layout="wide")
st.title("AIPulse Admin Console")
st.caption("Protected operational actions. Training requests are audited by actor and outcome.")

api_url = st.sidebar.text_input(
    "API URL", value=os.getenv("AIPULSE_API_URL", "http://localhost:8000")
)
access_token = st.sidebar.text_input(
    "Admin or ML engineer token",
    type="password",
    help="A short-lived JWT with the admin or ml_engineer role is required.",
)

if not access_token:
    st.warning("Enter an authorized token before performing administrative actions.")
    st.stop()

try:
    client = APIClient(api_url, access_token)
    models_response = client.list_models()
except (APIError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Train and register a model")
with st.form("train-model", clear_on_submit=False):
    left, right = st.columns(2)
    with left:
        dataset_name = st.text_input(
            "Processed dataset filename",
            placeholder="sample.processed.parquet",
            help="The file must already exist under the API's data/processed directory.",
        )
        model_name = st.text_input("Model name", value="ai_model")
        label_column = st.text_input("Label column", value="label")
    with right:
        model_type = st.selectbox(
            "Estimator",
            options=["logistic_regression", "random_forest"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        validation_fraction = st.slider(
            "Validation fraction", min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )
    confirmation = st.checkbox(
        "I understand this starts a training job and creates a model artifact."
    )
    submitted = st.form_submit_button("Start training", type="primary")

if submitted and not dataset_name:
    st.error("Enter a processed dataset filename before starting training.")
elif submitted and not confirmation:
    st.error("Confirm that you understand a model artifact will be created.")
elif submitted:
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "model_type": model_type,
        "label_column": label_column,
        "validation_fraction": validation_fraction,
    }
    with st.spinner("Training, evaluating, and registering the model…"):
        try:
            result = client.train_model(payload)
        except APIError as exc:
            if exc.status_code in {401, 403}:
                st.error("Your token does not authorize model training.")
            else:
                st.error(str(exc))
        else:
            validation = result.get("validation", {})
            registration = f"Registered {result.get('model')} version {result.get('version')}"
            if validation.get("passed"):
                st.success(f"{registration} as validated.")
            else:
                st.warning(f"{registration} as a candidate.")
                for failure in validation.get("failures", []):
                    st.write(f"Release gate: {failure}")
            st.json(
                {
                    "metrics": result.get("metrics"),
                    "checksum_sha256": result.get("checksum_sha256"),
                    "training_rows": result.get("training_rows"),
                    "validation_rows": result.get("validation_rows"),
                }
            )
            try:
                models_response = client.list_models()
            except APIError:
                st.warning("Training succeeded, but the registry could not be refreshed.")

st.subheader("Registry")
items = models_response.get("items", [])
if not items:
    st.info("The registry is empty.")
else:
    registry_rows = [
        {
            "Model": item.get("model_name"),
            "Version": item.get("version"),
            "Tags": ", ".join(item.get("tags", [])),
            "Dataset": item.get("training_data_version"),
            "Created": item.get("created_at"),
        }
        for item in items
    ]
    st.dataframe(pd.DataFrame(registry_rows), use_container_width=True, hide_index=True)
