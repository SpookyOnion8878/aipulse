"""Compatibility imports for the canonical data engineering pipeline."""

from ..data_engineering.data_pipeline import (
    DataPipeline,
    DataPipelineConfig,
    DataQualityReport,
    DataValidationError,
    main,
)

__all__ = [
    "DataPipeline",
    "DataPipelineConfig",
    "DataQualityReport",
    "DataValidationError",
    "main",
]


if __name__ == "__main__":
    main()
