"""
Data Pipeline — Step 3: Data Engineering

AIPulse Enterprise AI Platform — Data Pipeline Layer
Handles second-stage data processing for model training data.
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

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, 'data_pipeline.log')),
    ],
)
logger = logging.getLogger("aipulse.data_pipeline")


class DataPipeline:
    """Data processing pipeline for model training."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_sources = []
        self.transformations = []

    def load_data(self, source: str, filepath: str) -> pd.DataFrame:
        """Load data from various sources."""
        ext = Path(filepath).suffix.lower()

        if ext == ".csv":
            df = pd.read_csv(filepath, parse_dates=True)
            logger.info(f"Loaded CSV: {filepath} ({len(df)} rows)")
        elif ext in (".json", ".jsonl"):
            df = pd.read_json(filepath, orient='records')
            logger.info(f"Loaded JSON: {filepath} ({len(df)} rows)")
        elif ext == ".parquet":
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded Parquet: {filepath} ({len(df)} rows)")
        elif ext == ".pandas":
            df = pd.read_parquet(filepath)
            logger.info(f"Loaded Pandas: {filepath} ({len(df)} rows)")
        else:
            raise ValueError(f"Unsupported format: {ext}")

        self.data_sources.append({
            "source": source,
            "file": filepath,
            "rows": len(df),
            "columns": list(df.columns),
            "type": ext,
        })

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to the data."""
        # 1. Handle missing values
        df = self._handle_missing(df)

        # 2. Handle duplicates
        df = self._handle_duplicates(df)

        # 3. Remove outliers
        df = self._handle_outliers(df)

        # 4. Normalize numerical columns
        df = self._normalize(df)

        # 5. Encode categorical variables
        df = self._encode_categorical(df)

        # 6. Create derived features
        df = self._create_features(df)

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64', 'float32']:
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled missing values in '{col}' with median")
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Filled missing values in '{col}' with mode")
        return df

    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        logger.info(f"Removed {before - after} duplicate rows")
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df[col] < lower) | (df[col] > upper)
            n_outliers = outliers.sum()
            if n_outliers > 0:
                df.loc[outliers, col] = np.nan
                logger.info(f"Replaced {n_outliers} outliers in '{col}' with NaN")
        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical columns to [0, 1] range."""
        for col in df.select_dtypes(include=['number']).columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                logger.info(f"Normalized column '{col}'")
        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        for col in df.select_dtypes(include=['object', 'category']).columns:
            n_unique = df[col].nunique()
            if n_unique <= 20:
                df[col] = df[col].astype('category').cat.codes
                logger.info(f"Encoded '{col}' using label encoding ({n_unique} unique values)")
            else:
                df[col] = df[col].astype('category').cat.codes
                logger.info(f"Encoded '{col}' using label encoding (high cardinality)")
        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features."""
        # Convert date columns
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                logger.info(f"Created features from '{col}'")

        # Create interaction features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                df[f"{col1}_with_{col2}"] = df[col1] * df[col2]
                logger.info(f"Created interaction feature: {col1} * {col2}")

        return df

    def run(self, source: str, filepath: str, output_path: str = "data/processed") -> pd.DataFrame:
        """Run the complete data pipeline."""
        logger.info(f"Starting pipeline: {source} -> {filepath}")

        # Load
        df = self.load_data(source, filepath)

        # Transform
        df = self.transform(df)

        # Store
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{Path(filepath).stem}.processed.parquet")
        df.to_parquet(output_file, index=False)
        logger.info(f"Pipeline complete. Saved to {output_file}")

        return df


# Entry point
def main():
    """Run the data pipeline."""
    pipeline = DataPipeline(
        config={
            "source": "data/raw",
            "output": "data/processed",
            "batch_size": 1000,
        }
    )

    # Process sample data
    sample_df = pd.DataFrame({
        "id": range(1000),
        "value1": np.random.randn(1000),
        "value2": np.random.randn(1000),
        "category": np.random.choice(["A", "B", "C"], 1000),
        "label": np.random.choice([0, 1], 1000),
        "date": pd.date_range("2024-01-01", periods=1000),
    })
    os.makedirs("data/raw", exist_ok=True)
    sample_df.to_csv("data/raw/sample.csv", index=False)

    result = pipeline.run("sample", "data/raw/sample.csv")
    print(f"Pipeline complete. Processed {len(result)} rows.")


if __name__ == "__main__":
    main()
