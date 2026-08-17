from __future__ import annotations

from pathlib import Path

from src.system.model_registry import ModelRegistry, sha256_file


def register(registry: ModelRegistry, artifact: Path, version: str) -> None:
    registry.register(
        "classifier",
        version,
        {"accuracy": 0.9},
        {"model": str(artifact)},
        checksum_sha256=sha256_file(artifact),
    )


def test_registry_uses_semantic_version_order_and_cleans_index(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry.json")
    register(registry, artifact, "1.2.10")
    register(registry, artifact, "1.10.0")

    assert registry.get_model("classifier")["version"] == "1.10.0"
    assert registry.delete_model("classifier", "1.10.0") is True
    assert [item["version"] for item in registry.list_models()] == ["1.2.10"]


def test_registry_rejects_duplicate_version(tmp_path: Path) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"model")
    registry = ModelRegistry(tmp_path / "registry.json")
    register(registry, artifact, "1.0.0")
    try:
        register(registry, artifact, "1.0.0")
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Duplicate version should be rejected")
