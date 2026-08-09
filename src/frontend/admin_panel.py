"""
Admin Panel — Step 5: Frontend

AIPulse Enterprise AI Platform — Admin Panel
Role-based management and team administration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import hashlib

# Page config
st.set_page_config(
    page_title="Admin Panel",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("🔧 Admin Panel")

st.sidebar.markdown("**Role Management**")
roles = st.sidebar.multiselect("Select Roles", ["Admin", "Data Scientist", "ML Engineer", "DevOps", "Viewer"], default=["Admin"])

st.sidebar.markdown("**Team Management**")
team_size = st.sidebar.slider("Team Size", 1, 100, 10)
team_members = [f"Member_{i}" for i in range(team_size)]

st.sidebar.markdown("**License**")
license = st.sidebar.selectbox("License", ["Business", "Enterprise", "Free"])

st.sidebar.markdown("**Notifications**")
notifications = st.sidebar.checkbox("Enable Notifications", value=True)

st.sidebar.markdown("**Settings**")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
lang = st.sidebar.selectbox("Language", ["English", "Mandarin", "French"])

# Welcome Banner
st.markdown("# 🛠️ Admin Panel")
st.markdown("### Role-Based Access Control for AIPulse Platform")

# Dashboard Overview
st.header("📊 Overview")

# Stats
total_users = team_size
total_models = 156
total_pipelines = 89
total_issues = 12

col1, col2, col3 = st.columns(3)
col1.metric("Active Users", f"{total_users} ({role_distribution})")
col2.metric("Total Models", f"{total_models}")
col3.metric("Active Pipelines", f"{total_pipelines}")

# Role Distribution
st.header("👥 Role Distribution")
role_data = {
    "Admin": 2,
    "Data Scientist": 15,
    "ML Engineer": 30,
    "DevOps": 12,
    "Viewer": 45,
}
role_df = pd.DataFrame(list(role_data.items()), columns=["Role", "Count"])
st.dataframe(role_df, use_container_width=True)

# Model Management
st.header("🚀 Model Management")

model_df = pd.DataFrame({
    "Model Name": [f"Model_{i}" for i in range(10)],
    "Type": np.random.choice(["LSTM", "Transformer", "Random Forest"], 10),
    "Status": ["Active" if np.random.random() > 0.2 else "Inactive" for _ in range(10)],
    "Accuracy": np.random.uniform(0.85, 0.99, 10).round(4),
    "Last Updated": pd.date_range("2024-01-01", periods=10)[:10],
})

st.dataframe(model_df, use_container_width=True)

# Team Management
st.header("👥 Team Management")

# Group by role
grouped = model_df.groupby("Type").size().reset_index(name="Count")
st.dataframe(grouped, use_container_width=True)

# Model Registry
st.header("📦 Model Registry")

model_registry_df = pd.DataFrame({
    "Model": ["Model_1", "Model_2", "Model_3"],
    "Version": ["1.0.0", "1.1.0", "1.2.0"],
    "Status": ["Deployed", "Deployed", "In Progress"],
    "Accuracy": [0.92, 0.94, 0.89],
    "Environment": ["Production", "Staging", "Development"],
})
st.dataframe(model_registry_df, use_container_width=True)

# System Health
st.header("🏥 System Health")

health_df = pd.DataFrame({
    "Service": ["API Gateway", "Model Serving", "Feature Store", "Monitoring", "Database", "Storage"],
    "Status": ["🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟢 Healthy", "🟢 Healthy"],
    "CPU Usage": ["23%", "15%", "35%", "12%", "40%", "60%"],
    "Memory Usage": ["45%", "32%", "55%", "28%", "42%", "58%"],
    "Uptime": ["99.99%", "99.95%", "99.98%", "99.99%", "99.90%", "99.99%"],
})
st.dataframe(health_df, use_container_width=True)

# Alerts
st.header("🚨 Alerts & Notifications")

alert_df = pd.DataFrame({
    "Alert": ["High latency", "Model drift", "Deployment failed", "Health check failed"],
    "Severity": ["Warning", "Warning", "Critical", "Info"],
    "Message": [
        "Model inference latency > 200ms",
        "Model distribution shift detected",
        "Deployment failed due to timeout",
        "Health check failed for Model_3",
    ],
    "Time": ["2 min ago", "15 min ago", "1 hour ago", "3 hours ago"],
})
st.dataframe(alert_df, use_container_width=True)

# Audit
st.header("📋 Audit Trail")
st.info("All model deployments, access changes, and data updates are logged here.")

# Footer
st.markdown("---")
st.markdown("### 📘 AIPulse Enterprise AI Platform v1.0.0")
st.markdown(f"**Last Updated**: {datetime.now().isoformat()}")
