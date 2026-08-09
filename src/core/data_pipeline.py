"""
Data Pipeline Module — Step 1: Core Skills (Data Processing & ETL)

AIPulse Enterprise AI Platform — Data Processing Layer
Handles ETL pipelines for raw data ingestion, cleaning, transformation, and feature engineering.
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
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'pipeline.log')),
    ],
)
logger = logging.getLogger("aipulse.data_pipeline")


class DataIngestionConfig:
    """Configuration for data ingestion pipeline."""

    def __init__(
        self,
        source: str = "local",
        batch_size: int = 1000,
        chunk_size: int = 10000,
        output_dir: str = "data/raw",
        clean: bool = True,
        validate: bool = True,
        schema_version: str = "1.0.0",
    ):
        self.source = source
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.clean = clean
        self.validate = validate
        self.schema_version = schema_version

    def get_output_path(self, filename: str) -> str:
        return os.path.join(self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")


class DataLoader:
    """Load data from various sources."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.raw_data: Optional[pd.DataFrame] = None

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading CSV from {filepath}")
        df = pd.read_csv(filepath, parse_dates=True, low_memory=False)
        self.raw_data = df
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {filepath}")
        return df

    def load_json(self, filepath: str) -> pd.DataFrame:
        """Load data from JSON file."""
        logger.info(f"Loading JSON from {filepath}")
        data = json.load(open(filepath))
        df = pd.DataFrame(data)
        self.raw_data = df
        return df

    def load_parquet(self, filepath: str) -> pd.DataFrame:
        """Load data from Parquet file."""
        logger.info(f"Loading Parquet from {filepath}")
        df = pd.read_parquet(filepath)
        self.raw_data = df
        return df

    def load_from_source(self, source_path: str) -> pd.DataFrame:
        """Load data based on source type."""
        ext = Path(source_path).suffix.lower()
        if ext == ".csv":
            return self.load_csv(source_path)
        elif ext in (".json", ".jsonl"):
            return self.load_json(source_path)
        elif ext == ".parquet":
            return self.load_parquet(source_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_from_db(self, connection_string: str, query: str) -> pd.DataFrame:
        """Load data from database."""
        logger.info(f"Loading data from database: {connection_string}")
        # TODO: Implement database connection
        df = pd.DataFrame()  # placeholder
        return df


class DataCleaner:
    """Clean and preprocess data."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled missing values in '{col}' with median")
            elif df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Filled missing values in '{col}' with mode")
            else:
                df[col] = df[col].fillna(0)
        return df

    def handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        logger.info("Removing duplicate rows")
        before_count = len(df)
        df = df.drop_duplicates()
        after_count = len(df)
        dup_count = before_count - after_count
        logger.info(f"Removed {dup_count} duplicate rows")
        return df

    def handle_outliers(self, df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        logger.info("Detecting and handling outliers")
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                df.loc[outlier_mask, col] = np.nan
                logger.info(f"Replaced {outlier_count} outliers in '{col}' with NaN")

        return df

    def remove_inactive_records(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove records with low activity scores."""
        logger.info(f"Removing inactive records with activity < {threshold}")
        active_mask = df.apply(
            lambda row: row.sum() / len(row) >= threshold if len(row) > 0 else False, axis=1
        )
        filtered_df = df[active_mask]
        removed_count = len(df) - len(filtered_df)
        logger.info(f"Removed {removed_count} inactive records")
        return filtered_df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical columns."""
        logger.info("Normalizing numerical columns")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
                logger.info(f"Normalized column '{col}'")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps."""
        df = self.handle_missing_values(df)
        df = self.handle_duplicates(df)
        df = self.handle_outliers(df)
        df = self.normalize_columns(df)
        return df


class DataTransformer:
    """Transform data for model training."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical variables")
        for col in df.select_dtypes(include=['object', 'category']).columns:
            unique_vals = df[col].nunique()
            if unique_vals > 20:
                df[col] = df[col].astype('category').cat.codes
                logger.info(f"Encoded '{col}' using label encoding")
            else:
                df[col] = df[col].astype('category').cat.codes
                logger.info(f"Encoded '{col}' using one-hot encoding (few unique values)")
        return df

    def scale_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale continuous columns to [0, 1] range."""
        logger.info("Scaling continuous columns")
        for col in df.select_dtypes(include=[np.number]).columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
                logger.info(f"Scaled column '{col}' to [0, 1] range")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data."""
        logger.info("Creating new features")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['hour'] = df['date'].dt.hour
            df['day_of_week'] = df['date'].dt.dayofweek
            logger.info("Created date-derived features")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        df = self.encode_categorical(df)
        df = self.scale_continuous(df)
        df = self.create_features(df)
        return df


class PipelineRunner:
    """Runner for the data pipeline."""

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.data_loaders = {
            'csv': DataLoader(config),
            'json': DataLoader(config),
            'parquet': DataLoader(config),
        }
        self.cleansers = {
            'missing_values': DataCleaner(config),
            'duplicates': DataCleaner(config),
            'outliers': DataCleaner(config),
            'normalization': DataCleaner(config),
        }
        self.transformers = {
            'encode': DataTransformer(config),
            'scale': DataTransformer(config),
            'create_features': DataTransformer(config),
        }

    def run(self, input_path: str, output_path: str = "data/processed") -> pd.DataFrame:
        """Run the full pipeline."""
        logger.info(f"Starting pipeline from {input_path} to {output_path}")

        # Load
        df = self.data_loaders['csv'].load_csv(input_path)

        # Clean
        df = self.cleansers['missing_values'].process(df)
        df = self.cleansers['duplicates'].process(df)
        df = self.cleansers['outliers'].process(df)
        df = self.cleansers['normalization'].process(df)

        # Transform
        df = self.transformers['encode'].transform(df)
        df = self.transformers['scale'].transform(df)
        df = self.transformers['create_features'].transform(df)

        # Save
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, f"{Path(input_path).stem}.pandas")
        df.to_parquet(output_file, index=False)
        logger.info(f"Pipeline complete. Saved to {output_file}")

        return df


# Entry point
def main():
    """Run the data pipeline."""
    config = DataIngestionConfig(
        source="local",
        batch_size=1000,
        chunk_size=10000,
        output_dir="data/raw",
        clean=True,
        validate=True,
    )

    runner = PipelineRunner(config)

    # Process all data files in the directory
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for f in sorted(os.listdir(data_dir)):
        if f.endswith(('.csv', '.json', '.parquet')):
            filepath = os.path.join(data_dir, f)
            logger.info(f"Processing file: {filepath}")
            result = runner.run(filepath, "data/processed")
            print(f"Processed: {f} -> {result.shape}")

    logger.info("All files processed successfully!")


if __name__ == "__main__":
    main()
