"""
Frontend Dashboard — Step 5: Frontend

AIPulse Enterprise AI Platform — Dashboard Layer
Real-time dashboards for monitoring, management, and reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import os

# Page config
st.set_page_config(
    page_title="AIPulse Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("# ⚡ AIPulse Enterprise AI Platform")
st.markdown("### Real-time AI Operations Dashboard")

# Sidebar
st.sidebar.title("🔧 Configuration")
st.sidebar.markdown("**Model Configuration**")
model_type = st.sidebar.selectbox("Model Type", ["LSTM", "Transformer", "BERT", "Random Forest", "Gradient Boosting"])
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001)
batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)

st.sidebar.markdown("**Environment**")
env = st.sidebar.selectbox("Environment", ["Development", "Staging", "Production"])

st.sidebar.markdown("**Monitoring**")
auto_scale = st.sidebar.checkbox("Auto-scale Inference", value=True)
health_check = st.sidebar.checkbox("Health Check", value=True)

# Main content
st.title("📊 Dashboard Overview")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

kpi_data = [
    {"label": "Total Models", "value": "156", "color": "blue"},
    {"label": "Active Models", "value": "128", "color": "green"},
    {"label": "Avg Latency", "value": "124ms", "color": "orange"},
    {"label": "Uptime", "value": "99.97%", "color": "red"},
]

for i, (kpi, data) in enumerate([
    ("Total Models", "156", "blue"),
    ("Active Models", "128", "green"),
    ("Avg Latency", "124ms", "orange"),
    ("Uptime", "99.97%", "red"),
]):
    col = st.columns(4)[i]
    col.metric(kpi, data)

# Model Performance Chart
st.header("📈 Model Performance")

# Generate random model performance data
num_models = 10
model_names = [f"Model_{i}" for i in range(num_models)]
accuracy = np.random.uniform(0.85, 0.99, num_models)
precision = np.random.uniform(0.85, 0.99, num_models)
recall = np.random.uniform(0.80, 0.98, num_models)
latency = np.random.uniform(50, 500, num_models)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Accuracy & Precision")
    st.bar_chart(accuracy.values.reshape(-1, 1))

with col_b:
    st.subheader("Recall & Latency")
    st.bar_chart(recall.values.reshape(-1, 1))

# Model Distribution
st.header("🔍 Model Distribution")

dist_data = pd.DataFrame({
    "Model": model_names,
    "Accuracy": np.random.uniform(0.85, 0.99, num_models).round(4),
    "Latency": np.random.uniform(50, 500, num_models).round(1),
})

st.dataframe(dist_data, use_container_width=True)

# Model Ranking
st.header("🏆 Model Ranking")

ranking_data = pd.DataFrame({
    "Rank": [f"{i+1}" for i in range(num_models)],
    "Model": [f"Model_{i}" for i in range(num_models)],
    "Score": np.random.uniform(0.85, 0.99, num_models).round(4),
    "Status": ["🟢 Active" if np.random.random() > 0.15 else "🟡 Warning" for _ in range(num_models)],
})

st.dataframe(ranking_data, use_container_width=True)

# Live Inference
st.header("⚡ Live Inference")

inference_data = pd.DataFrame({
    "Inference": [f"Inference_{i}" for i in range(5)],
    "Latency (ms)": np.random.uniform(50, 500, 5).round(1),
    "Status": ["✅ Success" if np.random.random() > 0.15 else "❌ Error" for _ in range(5)],
})

st.dataframe(inference_data, use_container_width=True)

# Pipeline Status
st.header("🔄 Pipeline Status")

pipeline_data = pd.DataFrame({
    "Pipeline": ["ETL", "Training", "Evaluation", "Deployment", "Monitoring"],
    "Status": ["✅ Running", "✅ Completed", "✅ Completed", "🟢 Pending", "🟡 Running"],
    "Duration (s)": [30, 150, 45, 0, 25],
    "Models": ["model_1", "model_2", "model_1", "model_1", "model_1"],
})

st.dataframe(pipeline_data, use_container_width=True)

# Alerts
st.header("🚨 Alerts")

alerts = [
    {"model": "model_1", "type": "Warning", "message": "High latency on inference endpoint"},
    {"model": "model_2", "type": "Info", "message": "New model version deployed"},
    {"model": "model_3", "type": "Critical", "message": "Data drift detected in training set"},
]

alerts_df = pd.DataFrame(alerts)
st.dataframe(alerts_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📋 System Information")

st.markdown(f"**Platform**: AIPulse Enterprise AI Platform")
st.markdown(f"**Version**: 1.0.0")
st.markdown(f"**Environment**: {env}")
st.markdown(f"**Uptime**: {time.strftime('%H:%M:%S')}")
st.markdown(f"**Total Models**: {num_models}")
st.markdown(f"**Last Updated**: {datetime.now().isoformat()}")

# Health check
if health_check:
    health_data = {
        "Model Service": "✅ Healthy",
        "Feature Store": "✅ Healthy",
        "API Gateway": "✅ Healthy",
        "Database": "✅ Healthy",
        "Monitoring": "✅ Healthy",
        "Storage": "✅ Healthy",
        "Network": "✅ Healthy",
    }

    health_df = pd.DataFrame(list(health_data.items()))
    st.dataframe(health_df, use_container_width=True)
