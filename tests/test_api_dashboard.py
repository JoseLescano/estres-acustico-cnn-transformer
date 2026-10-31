from fastapi.testclient import TestClient
from pathlib import Path

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_dashboard_index_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "dashboard" in response.text.lower()


def test_dashboard_data_has_history():
    response = client.get("/dashboard-data")
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["current_stress"] <= 1
    assert len(data["history"]) >= 1


def test_predict_rejects_non_wav_file():
    response = client.post("/predict", files={"file": ("sample.txt", b"not-audio")})
    assert response.status_code == 415
    assert response.json()["detail"] == "El archivo debe ser .wav"


def test_predict_demo_wav():
    audio_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "demo_high_stress_1.wav"
    response = client.post(
        "/predict",
        files={"file": (audio_path.name, audio_path.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "predicted"
    assert 0 <= data["stress_index"] <= 1
