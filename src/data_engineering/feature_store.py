"""Safe local feature-store adapter for the single-node deployment."""

from __future__ import annotations

import os
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import pandas as pd

FEATURE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class FeatureStore:
    """Atomic Parquet store with a bounded in-process read cache."""

    def __init__(self, offline_path: str | Path = "features/offline", max_cache_items: int = 32):
        self.offline_path = Path(offline_path).resolve()
        self.max_cache_items = max_cache_items
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._lock = threading.RLock()

    def _path(self, feature_name: str) -> Path:
        if FEATURE_NAME.fullmatch(feature_name) is None:
            raise ValueError("Invalid feature name")
        path = (self.offline_path / f"{feature_name}.parquet").resolve()
        if not path.is_relative_to(self.offline_path):
            raise ValueError("Feature path escapes the store")
        return path

    def store_offline_features(self, feature_name: str, frame: pd.DataFrame) -> Path:
        if frame.empty:
            raise ValueError("Feature frame must not be empty")
        path = self._path(feature_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{feature_name}.", suffix=".parquet", dir=path.parent
        )
        os.close(descriptor)
        try:
            frame.to_parquet(temp_name, index=False)
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        with self._lock:
            self._cache[feature_name] = frame.copy(deep=True)
            self._cache.move_to_end(feature_name)
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        return path

    def load_offline_features(self, feature_name: str) -> pd.DataFrame:
        path = self._path(feature_name)
        if not path.is_file():
            raise FileNotFoundError(f"Unknown feature set: {feature_name}")
        return pd.read_parquet(path)

    def get_online_feature(self, feature_name: str) -> pd.DataFrame | None:
        with self._lock:
            frame = self._cache.get(feature_name)
            if frame is None:
                return None
            self._cache.move_to_end(feature_name)
            return frame.copy(deep=True)

    def update_online_feature(self, feature_name: str, frame: pd.DataFrame) -> None:
        self._path(feature_name)
        with self._lock:
            self._cache[feature_name] = frame.copy(deep=True)
            self._cache.move_to_end(feature_name)
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)

    def create_feature_vector(self, frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        missing = sorted(set(feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Unknown features: {missing}")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("feature_names contains duplicates")
        return frame.loc[:, feature_names].copy()

    def feature_store_pipeline(
        self, source_frame: pd.DataFrame, feature_names: list[str], *, name: str = "features"
    ) -> pd.DataFrame:
        vector = self.create_feature_vector(source_frame, feature_names)
        self.store_offline_features(name, vector)
        return vector
