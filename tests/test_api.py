from __future__ import annotations

from fastapi.testclient import TestClient

from src.security.auth import create_access_token
from src.system.api_gateway import create_app


def authorization(settings, roles: list[str]) -> dict[str, str]:
    token = create_access_token(settings, "test-user", roles)
    return {"Authorization": f"Bearer {token}"}


def test_health_and_authentication(settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["authentication_configured"] is True

        unauthenticated = client.get("/api/v1/models")
        assert unauthenticated.status_code == 401

        viewer = client.get("/api/v1/models", headers=authorization(settings, ["viewer"]))
        assert viewer.status_code == 200
        assert viewer.json() == {"items": [], "count": 0}


def test_training_endpoint_creates_real_artifact(settings) -> None:
    rows = ["x,category,label"]
    rows.extend(f"{index},{'a' if index % 2 else 'b'},{index % 2}" for index in range(40))
    dataset = settings.processed_data_dir / "training.csv"
    dataset.write_text("\n".join(rows), encoding="utf-8")

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/models/train",
            headers=authorization(settings, ["ml_engineer"]),
            json={
                "dataset_name": "training.csv",
                "model_name": "classifier",
                "model_type": "logistic_regression",
                "label_column": "label",
                "validation_fraction": 0.25,
            },
        )
        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["model"] == "classifier"
        assert len(payload["checksum_sha256"]) == 64
        assert (settings.models_dir / "classifier" / payload["version"] / "model.joblib").is_file()


def test_training_requires_privileged_role(settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/models/train",
            headers=authorization(settings, ["viewer"]),
            json={"dataset_name": "training.csv"},
        )
    assert response.status_code == 403
