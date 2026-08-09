"""
Model Registry — Step 2: System Design (Enterprise)

AIPulse Enterprise AI Platform — Model Registry Layer
Manages model versions, tracking, and deployment lifecycle.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import yaml

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, 'model_registry.log')),
    ],
)
logger = logging.getLogger("aipulse.model_registry")


class ModelVersion:
    """Represents a version of a model."""

    def __init__(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
        created_at: str = None,
        tags: List[str] = None,
        training_data_version: str = None,
    ):
        self.model_name = model_name
        self.version = version
        self.metrics = metrics
        self.artifacts = artifacts
        self.created_at = created_at or datetime.now().isoformat()
        self.tags = tags or []
        self.training_data_version = training_data_version

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "tags": self.tags,
            "training_data_version": self.training_data_version,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        """Create ModelVersion from dictionary."""
        return cls(
            model_name=data.get("model_name", ""),
            version=data.get("version", ""),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", {}),
            created_at=data.get("created_at", ""),
            tags=data.get("tags", []),
            training_data_version=data.get("training_data_version", ""),
        )


class ModelRegistry:
    """Registry for managing model versions."""

    def __init__(
        self,
        registry_path: str = "models/registry.json",
        storage_path: str = "models/storage",
    ):
        self.registry_path = registry_path
        self.storage_path = storage_path
        self.registry_data = self._load_registry()
        self.storage_exists = os.path.exists(self.storage_path)

    def _load_registry(self) -> Dict:
        """Load registry from file."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}, "versions": [], "metadata": {}}

    def _save_registry(self) -> None:
        """Save registry to file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry_data, f, indent=2)

    def register(self, model_name: str, version: str, metrics: Dict, artifacts: Dict, tags: List[str] = None) -> str:
        """Register a new model version."""
        logger.info(f"Registering model: {model_name} v{version}")

        # Check if version already exists
        if model_name in self.registry_data["models"]:
            existing_version = f"{model_name}:{version}"
            if existing_version in self.registry_data["versions"]:
                logger.info(f"Model {model_name} v{version} already exists, skipping")
                return version

        # Create new version
        version_obj = ModelVersion(
            model_name=model_name,
            version=version,
            metrics=metrics,
            artifacts=artifacts,
            tags=tags or [],
        )

        # Update registry
        if model_name not in self.registry_data["models"]:
            self.registry_data["models"][model_name] = {}

        self.registry_data["models"][model_name][version] = version_obj.to_dict()
        self.registry_data["versions"].append(f"{model_name}:{version}")

        # Save
        self._save_registry()

        logger.info(f"Model {model_name} v{version} registered successfully")
        return version

    def get_model(self, model_name: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get a model by name and version."""
        if model_name not in self.registry_data["models"]:
            return None

        if version:
            return self.registry_data["models"].get(model_name, {}).get(version)

        # Return the latest version
        versions = self.registry_data["models"].get(model_name, {})
        if not versions:
            return None
        latest_version = max(versions.keys(), key=lambda v: self._version_to_int(v))
        return self.registry_data["models"][model_name].get(latest_version)

    def _version_to_int(self, version: str) -> int:
        """Convert version string to integer for sorting."""
        parts = version.split(".")
        return int("".join(parts)) if parts else 0

    def list_models(self, filter_model: Optional[str] = None) -> List[Dict]:
        """List all models."""
        if filter_model:
            return [
                v for v in self.registry_data["versions"]
                if v.startswith(f"{filter_model}:")
            ]
        return self.registry_data["versions"]

    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model version."""
        if model_name not in self.registry_data["models"]:
            return False

        if version:
            if version not in self.registry_data["models"][model_name]:
                return False
            del self.registry_data["models"][model_name][version]
        else:
            # Delete all versions
            self.registry_data["models"].pop(model_name, None)

        self._save_registry()
        return True

    def get_all_metrics(self) -> List[Dict]:
        """Get all metrics across all models."""
        all_metrics = []
        for model_name, versions in self.registry_data["models"].items():
            for version, data in versions.items():
                all_metrics.append({
                    "model_name": model_name,
                    "version": version,
                    "metrics": data.get("metrics", {}),
                })
        return all_metrics

    def get_model_stats(self, model_name: str) -> Dict:
        """Get statistics for a model."""
        versions = self.registry_data["models"].get(model_name, {})
        if not versions:
            return {"model_name": model_name, "total_versions": 0, "versions": []}

        total_versions = len(versions)
        versions_list = list(versions.keys())
        latest_version = versions_list[-1] if versions_list else ""

        return {
            "model_name": model_name,
            "total_versions": total_versions,
            "latest_version": latest_version,
            "versions": versions_list,
        }


# Entry point
def main():
    """Run model registry."""
    registry = ModelRegistry(registry_path="models/registry.json", storage_path="models/storage")

    # Register a model
    metrics = {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92,
        "auc_roc": 0.95,
        "mse": 0.05,
    }

    version = registry.register(
        model_name="ai_model",
        version="1.0.0",
        metrics=metrics,
        artifacts={"checkpoint": "models/ai_model/checkpoint.pth", "config": "models/ai_model/config.json"},
        tags=["production", "lstm"],
    )

    print(f"Registered model: ai_model v{version}")

    # Get model
    model = registry.get_model("ai_model")
    print(f"Model info: {model}")

    # List all models
    all_models = registry.list_models()
    print(f"All models: {all_models}")

    # Get model stats
    stats = registry.get_model_stats("ai_model")
    print(f"Model stats: {stats}")


if __name__ == "__main__":
    main()
