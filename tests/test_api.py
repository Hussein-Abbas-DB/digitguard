import pytest
from fastapi.testclient import TestClient

from digitguard.api import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as app_client:
        yield app_client


def test_health_and_web_assets(client):
    assert client.get("/healthz").json()["status"] == "ready"
    assert "Can the model read it?" in client.get("/").text
    assert client.get("/static/lab.js").status_code == 200
    assert "/v1/predict" in client.get("/openapi.json").json()["paths"]


def test_real_model_serves_all_digits(client):
    accepted = 0
    for digit in range(10):
        sample = client.get("/v1/example", params={"digit": digit}).json()
        assert sample["digit"] == digit
        response = client.post("/v1/predict", json={"pixels": sample["pixels"]})
        assert response.status_code == 200
        result = response.json()
        assert len(result["probabilities"]) == 10
        assert sum(result["probabilities"]) == pytest.approx(1)
        if result["reason"] == "accepted":
            accepted += 1
            assert result["prediction"] in result["candidates"]
            assert result["prediction"] == digit
        else:
            assert result["prediction"] is None
    assert accepted >= 5


def test_blank_and_validation(client):
    blank = client.post("/v1/predict", json={"pixels": [0] * 64}).json()
    assert blank["reason"] == "blank_input"
    assert blank["prediction"] is None
    for pixels in ([0] * 63, [-1] * 64, [True] * 64, ["4"] * 64):
        assert client.post("/v1/predict", json={"pixels": pixels}).status_code == 422
    assert client.get("/v1/example?digit=10").status_code == 422


def test_missing_artifact_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Model missing"):
        with TestClient(create_app(tmp_path / "missing.joblib")):
            pass
