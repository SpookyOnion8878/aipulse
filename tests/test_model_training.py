from __future__ import annotations

from pathlib import Path

import pandas as pd
from aipulse.core.model_training import TrainingConfig, TrainingService
from aipulse.system.model_registry import ModelRegistry


def test_training_supports_datetime_features(settings) -> None:
    dataset = pd.DataFrame(
        {
            "event_at": pd.date_range("2026-01-01", periods=40, freq="h"),
            "amount": [float(index) for index in range(40)],
            "category": ["a" if index % 2 else "b" for index in range(40)],
            "label": [index % 2 for index in range(40)],
        }
    )
    dataset_path = settings.processed_data_dir / "datetime.parquet"
    dataset.to_parquet(dataset_path, index=False)
    registry = ModelRegistry(settings.registry_path)

    result = TrainingService(settings, registry).train(
        dataset_path.name,
        TrainingConfig(model_name="datetime-model", random_state=7),
    )

    assert result["training_rows"] == 32
    assert result["validation_rows"] == 8
    assert Path(result["artifact"]).is_file()
    assert registry.get_model("datetime-model", result["version"]) is not None
