"""Deterministic data ingestion and validation pipeline.

Learned encoders and scalers deliberately live in the model pipeline so they
are fit on training data only and serialized with the model artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..logging_config import configure_logging


class DataValidationError(ValueError):
    """Raised when input data violates the pipeline contract."""


@dataclass(frozen=True, slots=True)
class DataPipelineConfig:
    label_column: str = "label"
    drop_duplicates: bool = True
    clip_numeric_outliers: bool = True
    outlier_iqr_multiplier: float = 3.0
    minimum_rows: int = 20


@dataclass(slots=True)
class DataQualityReport:
    source: str
    rows_before: int
    rows_after: int
    duplicates_removed: int
    filled_values: dict[str, int] = field(default_factory=dict)
    clipped_values: dict[str, int] = field(default_factory=dict)
    derived_date_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class DataPipeline:
    """Clean raw tabular data without fitting model-time transformations."""

    def __init__(self, config: DataPipelineConfig | None = None) -> None:
        self.config = config or DataPipelineConfig()
        self.last_report: DataQualityReport | None = None

    @staticmethod
    def load_data(path: str | Path) -> pd.DataFrame:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Data source not found: {source}")
        suffix = source.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix in {".parquet", ".pandas"}:
            return pd.read_parquet(source)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(source, lines=suffix == ".jsonl")
        raise DataValidationError(f"Unsupported data format: {suffix}")

    def _validate_input(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise DataValidationError("Dataset is empty")
        if len(frame) < self.config.minimum_rows:
            raise DataValidationError(
                f"Dataset has {len(frame)} rows; minimum is {self.config.minimum_rows}"
            )
        if not frame.columns.is_unique:
            raise DataValidationError("Dataset contains duplicate column names")
        if self.config.label_column not in frame.columns:
            raise DataValidationError(f"Missing label column: {self.config.label_column}")
        if frame[self.config.label_column].isna().any():
            raise DataValidationError("Label column contains missing values")
        all_null = frame.columns[frame.isna().all()].tolist()
        if all_null:
            raise DataValidationError(f"All-null columns are not allowed: {all_null}")

    def transform(self, frame: pd.DataFrame, *, source: str = "in-memory") -> pd.DataFrame:
        self._validate_input(frame)
        result = frame.copy(deep=True)
        rows_before = len(result)
        if self.config.drop_duplicates:
            result = result.drop_duplicates().reset_index(drop=True)
        report = DataQualityReport(
            source=source,
            rows_before=rows_before,
            rows_after=len(result),
            duplicates_removed=rows_before - len(result),
        )

        date_candidates = [
            column
            for column in result.select_dtypes(include=["object", "string"]).columns
            if "date" in column.lower() or "time" in column.lower()
        ]
        for column in date_candidates:
            parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
            valid_ratio = float(parsed.notna().mean())
            if valid_ratio < 0.95:
                report.warnings.append(
                    f"{column} was not expanded as a date ({valid_ratio:.1%} valid values)"
                )
                continue
            result[f"{column}_year"] = parsed.dt.year
            result[f"{column}_month"] = parsed.dt.month
            result[f"{column}_day"] = parsed.dt.day
            result[f"{column}_dayofweek"] = parsed.dt.dayofweek
            result.drop(columns=[column], inplace=True)
            report.derived_date_columns.append(column)

        feature_columns = [
            column for column in result.columns if column != self.config.label_column
        ]
        numeric_columns = result[feature_columns].select_dtypes(include="number").columns
        for column in numeric_columns:
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
            missing = int(result[column].isna().sum())
            if missing:
                result[column] = result[column].fillna(result[column].median())
                report.filled_values[column] = missing
            if self.config.clip_numeric_outliers:
                q1 = float(result[column].quantile(0.25))
                q3 = float(result[column].quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - self.config.outlier_iqr_multiplier * iqr
                    upper = q3 + self.config.outlier_iqr_multiplier * iqr
                    changed = int(((result[column] < lower) | (result[column] > upper)).sum())
                    if changed:
                        result[column] = result[column].clip(lower, upper)
                        report.clipped_values[column] = changed

        categorical_columns = [
            column for column in feature_columns if column not in numeric_columns
        ]
        for column in categorical_columns:
            missing = int(result[column].isna().sum())
            if missing:
                result[column] = result[column].fillna("__missing__")
                report.filled_values[column] = missing

        if result.isna().any().any():
            columns = result.columns[result.isna().any()].tolist()
            raise DataValidationError(f"Null values remain after cleaning: {columns}")
        report.rows_after = len(result)
        self.last_report = report
        return result

    def run(self, source_path: str | Path, output_dir: str | Path) -> Path:
        source = Path(source_path).resolve()
        frame = self.load_data(source)
        transformed = self.transform(frame, source=str(source))
        destination_dir = Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{source.stem}.processed.parquet"
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".parquet", dir=destination_dir
        )
        os.close(descriptor)
        try:
            transformed.to_parquet(temp_name, index=False)
            os.replace(temp_name, destination)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        if self.last_report is not None:
            report_path = destination.with_suffix(".quality.json")
            report_path.write_text(
                json.dumps(asdict(self.last_report), indent=2, sort_keys=True), encoding="utf-8"
            )
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and validate a tabular dataset")
    parser.add_argument("source")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--label", default="label")
    args = parser.parse_args()
    configure_logging()
    pipeline = DataPipeline(DataPipelineConfig(label_column=args.label))
    destination = pipeline.run(args.source, args.output)
    print(destination)


if __name__ == "__main__":
    main()
