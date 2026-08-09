"""
Data Engineering — Step 3: Data Engineering

AIPulse Enterprise AI Platform — Data Engineering Layer
Handles ETL pipelines, feature stores, data versioning, and data quality.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, 'data_engineering.log')),
    ],
)
logger = logging.getLogger("aipulse.data_engineering")


class FeatureStore:
    """Feature store for storing and retrieving features."""

    def __init__(self, store_path: str = "features/online", offline_path: str = "features/offline"):
        self.store_path = store_path
        self.offline_path = offline_path
        self.online_store = {}
        self.offline_store = {}

    def load_offline_features(self, feature_name: str) -> pd.DataFrame:
        """Load offline features from Parquet file."""
        if not os.path.exists(self.offline_path):
            os.makedirs(self.offline_path, exist_ok=True)

        filepath = os.path.join(self.offline_path, f"{feature_name}.parquet")
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded offline features from {filepath}")
            return df
        return pd.DataFrame()

    def store_offline_features(self, feature_name: str, df: pd.DataFrame) -> None:
        """Store offline features to Parquet file."""
        os.makedirs(self.offline_path, exist_ok=True)
        filepath = os.path.join(self.offline_path, f"{feature_name}.parquet")
        df.to_parquet(filepath, index=False)
        logger.info(f"Stored offline features: {filepath}")

    def get_online_feature(self, feature_name: str) -> Optional[pd.DataFrame]:
        """Get feature from online store."""
        return self.online_store.get(feature_name, None)

    def update_online_feature(self, feature_name: str, df: pd.DataFrame) -> None:
        """Update online feature store."""
        self.online_store[feature_name] = df
        logger.info(f"Updated online feature: {feature_name}")

    def create_feature_vector(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Create feature vector from DataFrame."""
        features = df[feature_names]
        logger.info(f"Created feature vector with {len(features.columns)} features")
        return features

    def feature_store_pipeline(self, source_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Run feature store pipeline."""
        # Create feature vector
        feature_vector = self.create_feature_vector(source_df, feature_names)

        # Store offline
        self.store_offline_features("feature_store", feature_vector)

        # Update online store
        self.update_online_feature("feature_store", feature_vector)

        return feature_vector

    def save(self) -> None:
        """Save offline features to disk."""
        os.makedirs(self.offline_path, exist_ok=True)
        for feature_name, df in self.offline_store.items():
            filepath = os.path.join(self.offline_path, f"{feature_name}.parquet")
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved offline feature store: {filepath}")


class DataVersioning:
    """Version control for data using DVC."""

    def __init__(self, data_dir: str = "data/dvc", dvc_path: str = ".dvc"):
        self.data_dir = data_dir
        self.dvc_path = dvc_path
        self.config = {
            "version_control": "dvc",
            "remote": "s3://aipulse-data",
            "branch": "main",
        }

    def init_repo(self) -> None:
        """Initialize DVC repository."""
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Initializing DVC repository at {self.data_dir}")

    def add_data(self, file_path: str, version_name: str) -> str:
        """Add data to DVC."""
        os.makedirs(self.data_dir, exist_ok=True)

        # Simulate DVC operation
        output_path = os.path.join(self.data_dir, f"{version_name}.parquet")
        os.makedirs(self.data_dir, exist_ok=True)

        # Create versioned data
        logger.info(f"Added data to DVC: {file_path} -> {version_name}")
        return output_path

    def commit(self, version_name: str, message: str = "") -> None:
        """Commit data version to DVC."""
        # Simulate DVC commit
        logger.info(f"Committed data version: {version_name}")

    def tag_data(self, version_name: str, tag: str) -> None:
        """Tag a data version."""
        logger.info(f"Tagged data version {version_name} as {tag}")

    def list_versions(self) -> List[str]:
        """List all data versions."""
        return os.listdir(self.data_dir) if os.path.exists(self.data_dir) else []

    def restore_version(self, version_name: str) -> pd.DataFrame:
        """Restore data to a specific version."""
        filepath = os.path.join(self.data_dir, f"{version_name}.parquet")
        if os.path.exists(filepath):
            df = pd.read_parquet(filepath)
            logger.info(f"Restored data version: {version_name}")
            return df
        return pd.DataFrame()

    def diff(self, version_a: str, version_b: str) -> pd.DataFrame:
        """Compare two versions of data."""
        df_a = self.restore_version(version_a)
        df_b = self.restore_version(version_b)

        if df_a.empty or df_b.empty:
            return pd.DataFrame()

        # Compute difference
        diff = df_a.compare(df_b)
        logger.info(f"Data diff computed: {len(diff)} columns differ")
        return diff


class DataPipeline:
    """Data pipeline for ETL processing."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.steps = self._configure_steps()
        self.pipeline_log: List[Dict] = []

    def _configure_steps(self) -> List[Dict]:
        """Configure pipeline steps."""
        return [
            {
                "name": "ingest",
                "description": "Ingest raw data from source",
                "handler": "data_pipeline",
                "config": self.config,
            },
            {
                "name": "clean",
                "description": "Clean and preprocess data",
                "handler": "data_pipeline",
                "config": self.config,
            },
            {
                "name": "transform",
                "description": "Transform data for model training",
                "handler": "data_pipeline",
                "config": self.config,
            },
            {
                "name": "validate",
                "description": "Validate data quality",
                "handler": "data_pipeline",
                "config": self.config,
            },
            {
                "name": "store",
                "description": "Store processed data in feature store",
                "handler": "data_pipeline",
                "config": self.config,
            },
        ]

    def run(self, source_path: str, output_path: str = "data/processed") -> pd.DataFrame:
        """Run the full ETL pipeline."""
        logger.info(f"Starting ETL pipeline: {source_path} -> {output_path}")

        # Step 1: Ingest
        logger.info("Step 1/5: Ingesting data...")
        df = source_path if isinstance(source_path, pd.DataFrame) else pd.read_csv(source_path)
        self.pipeline_log.append({"step": "ingest", "status": "success", "rows": len(df)})

        # Step 2: Clean
        logger.info("Step 2/5: Cleaning data...")
        df = self._clean_data(df)
        self.pipeline_log.append({"step": "clean", "status": "success", "rows": len(df)})

        # Step 3: Transform
        logger.info("Step 3/5: Transforming data...")
        df = self._transform_data(df)
        self.pipeline_log.append({"step": "transform", "status": "success", "rows": len(df)})

        # Step 4: Validate
        logger.info("Step 4/5: Validating data...")
        df = self._validate_data(df)
        self.pipeline_log.append({"step": "validate", "status": "success", "rows": len(df)})

        # Step 5: Store
        logger.info("Step 5/5: Storing data...")
        self._store_data(df, output_path)
        self.pipeline_log.append({"step": "store", "status": "success", "rows": len(df)})

        logger.info(f"ETL pipeline complete. Processed {len(df)} rows.")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data."""
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle outliers
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].apply(
                lambda x: np.nan if (x < Q1 - 1.5 * IQR) or (x > Q3 + 1.5 * IQR) else x
            )

        return df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        # Normalize numeric columns
        for col in df.select_dtypes(include=['number']).columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)

        # Encode categoricals
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() <= 20:
                df[col] = df[col].astype('category').cat.codes

        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality."""
        # Check data types
        assert df.dtypes.apply(lambda x: x.kind in ['f', 'i', 'O']).all(), "Data types mismatch"

        # Fill any missing values left by outlier handling, then validate
        df = df.fillna(df.mean(numeric_only=True))

        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            assert (df[col] >= 0).all(), f"Negative values found in {col}"

        # Check for nulls
        assert df.isnull().sum().sum() == 0, "Null values found in data"

        return df

    def _store_data(self, df: pd.DataFrame, output_path: str) -> None:
        """Store data."""
        os.makedirs(output_path, exist_ok=True)
        filepath = os.path.join(output_path, "processed_data.parquet")
        df.to_parquet(filepath, index=False)
        logger.info(f"Data stored to {filepath}")


# Entry point
def main():
    """Run the data pipeline."""
    config = {
        "source": "data/raw",
        "output": "data/processed",
        "batch_size": 1000,
        "chunk_size": 10000,
    }

    pipeline = DataPipeline(config)

    # Process sample data
    sample_data = pd.DataFrame({
        "id": range(1000),
        "value1": np.abs(np.random.randn(1000)),
        "value2": np.abs(np.random.randn(1000)),
        "category": np.random.choice(["A", "B", "C"], 1000),
        "label": np.random.choice([0, 1], 1000),
    })

    result = pipeline.run(sample_data, "data/processed")
    print(f"Pipeline complete. Processed {len(result)} rows.")


if __name__ == "__main__":
    main()
