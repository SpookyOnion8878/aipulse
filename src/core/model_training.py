"""Deterministic tabular model training service."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..config import Settings
from ..logging_config import configure_logging
from ..system.model_registry import ModelRegistry, sha256_file
from .evaluation import EvaluationMetrics, ModelValidator


def _datetime_to_epoch_seconds(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime features to float epoch seconds while preserving missing values."""

    converted = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = pd.to_datetime(frame[column], errors="coerce", utc=True)
        numeric = values.astype("int64").astype("float64") / 1_000_000_000
        converted[column] = numeric.mask(values.isna(), np.nan)
    return converted


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    model_name: str = "ai_model"
    model_type: Literal["logistic_regression", "random_forest"] = "logistic_regression"
    label_column: str = "label"
    validation_fraction: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        if not 0.1 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be between 0.1 and 0.5")


class TrainingService:
    """Train, evaluate, serialize, and register a tabular classifier."""

    def __init__(self, settings: Settings, registry: ModelRegistry | None = None) -> None:
        self.settings = settings
        self.registry = registry or ModelRegistry(settings.registry_path)

    def _safe_dataset_path(self, dataset_name: str) -> Path:
        if Path(dataset_name).name != dataset_name:
            raise ValueError("dataset_name must be a filename, not a path")
        candidate = (self.settings.processed_data_dir / dataset_name).resolve()
        if not candidate.is_relative_to(self.settings.processed_data_dir.resolve()):
            raise ValueError("dataset path escapes the processed data directory")
        if candidate.suffix.lower() not in {".csv", ".parquet", ".pandas"}:
            raise ValueError("dataset must be CSV or Parquet")
        if not candidate.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")
        return candidate

    @staticmethod
    def _load_dataset(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    @staticmethod
    def _build_estimator(
        frame: pd.DataFrame,
        model_type: Literal["logistic_regression", "random_forest"],
        random_state: int,
    ) -> Pipeline:
        numeric_columns = frame.select_dtypes(include="number").columns.tolist()
        datetime_columns = [
            column
            for column in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[column].dtype)
        ]
        categorical_columns = [
            column
            for column in frame.columns
            if column not in numeric_columns and column not in datetime_columns
        ]
        if not numeric_columns and not datetime_columns and not categorical_columns:
            raise ValueError("Dataset has no usable feature columns")

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_columns:
            numeric = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric, numeric_columns))
        if datetime_columns:
            datetime = Pipeline(
                steps=[
                    (
                        "to_epoch_seconds",
                        FunctionTransformer(
                            _datetime_to_epoch_seconds,
                            validate=False,
                            feature_names_out="one-to-one",
                        ),
                    ),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("datetime", datetime, datetime_columns))
        if categorical_columns:
            categorical = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("categorical", categorical, categorical_columns))

        classifier: Any
        if model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=200,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        else:
            classifier = LogisticRegression(
                max_iter=1_000,
                random_state=random_state,
                class_weight="balanced",
            )
        return Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer(transformers=transformers)),
                ("classifier", classifier),
            ]
        )

    def train(self, dataset_name: str, config: TrainingConfig) -> dict[str, Any]:
        dataset_path = self._safe_dataset_path(dataset_name)
        data = self._load_dataset(dataset_path)
        if config.label_column not in data.columns:
            raise ValueError(f"Missing label column: {config.label_column}")
        if len(data) < 20:
            raise ValueError("At least 20 rows are required for a train/validation split")

        target = data[config.label_column]
        features = data.drop(columns=[config.label_column])
        if target.nunique(dropna=True) < 2:
            raise ValueError("Training requires at least two target classes")
        if target.isna().any():
            raise ValueError("Target column contains missing values")

        x_train, x_validation, y_train, y_validation = train_test_split(
            features,
            target,
            test_size=config.validation_fraction,
            random_state=config.random_state,
            stratify=target,
        )
        estimator = self._build_estimator(x_train, config.model_type, config.random_state)
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_validation)
        score = None
        if hasattr(estimator, "predict_proba") and target.nunique() == 2:
            score = estimator.predict_proba(x_validation)[:, 1]
        metrics = EvaluationMetrics().classification_report(y_validation, prediction, score)
        validation = ModelValidator().validate_model(metrics)

        version = f"1.0.{int(time.time())}"
        artifact_path = self.settings.models_dir / config.model_name / version / "model.joblib"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".model.", suffix=".joblib", dir=artifact_path.parent
        )
        os.close(descriptor)
        try:
            joblib.dump(estimator, temp_name)
            os.replace(temp_name, artifact_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

        checksum = sha256_file(artifact_path)
        record = self.registry.register(
            config.model_name,
            version,
            metrics,
            {"model": str(artifact_path)},
            checksum_sha256=checksum,
            tags=["candidate" if not validation["passed"] else "validated"],
            training_data_version=dataset_path.name,
        )
        return {
            "model": record.model_name,
            "version": record.version,
            "metrics": metrics,
            "validation": validation,
            "artifact": str(artifact_path),
            "checksum_sha256": checksum,
            "training_rows": len(x_train),
            "validation_rows": len(x_validation),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register an AIPulse model")
    parser.add_argument("dataset", help="Filename under data/processed")
    parser.add_argument("--label", default="label")
    parser.add_argument("--model-name", default="ai_model")
    parser.add_argument(
        "--model-type",
        choices=("logistic_regression", "random_forest"),
        default="logistic_regression",
    )
    args = parser.parse_args()
    configure_logging()
    settings = Settings.from_env()
    settings.ensure_runtime_directories()
    service = TrainingService(settings)
    result = service.train(
        args.dataset,
        TrainingConfig(
            model_name=args.model_name,
            model_type=args.model_type,
            label_column=args.label,
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
