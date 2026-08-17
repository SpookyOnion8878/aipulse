from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.data_pipeline import DataPipeline, DataPipelineConfig


def test_pipeline_preserves_target_and_derives_dates() -> None:
    frame = pd.DataFrame(
        {
            "value": [*range(24), np.nan],
            "category": ["a", "b"] * 12 + [None],
            "event_date": pd.date_range("2026-01-01", periods=25).astype(str),
            "label": [0, 1] * 12 + [1],
        }
    )
    original = frame.copy(deep=True)
    pipeline = DataPipeline(DataPipelineConfig(minimum_rows=20))
    result = pipeline.transform(frame)

    assert result["label"].tolist() == original["label"].tolist()
    assert "event_date" not in result
    assert "event_date_year" in result
    assert result.isna().sum().sum() == 0
    pd.testing.assert_frame_equal(frame, original)


def test_pipeline_does_not_create_target_interactions() -> None:
    frame = pd.DataFrame({"a": range(20), "b": range(20, 40), "label": [0, 1] * 10})
    result = DataPipeline().transform(frame)
    assert set(result.columns) == {"a", "b", "label"}
