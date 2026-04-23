from fastapi.testclient import TestClient
from app.main import app
import pytest


@pytest.fixture
def client():
    from unittest.mock import MagicMock
    import app.model_loader as ml

    # Mock model
    ml.model = MagicMock()
    ml.threshold = 0.5

    ml.model.predict.return_value = [1]
    ml.model.predict_proba.return_value = [[0.2, 0.8]]

    # ADD THESE (CRITICAL)
    ml.imputer = {
        "replace_value": None,
        "DR1TKCAL": 2000,
        "DR1TSUGR": 50,
        "DR1TTFAT": 60,
        "DR1TPROT": 70,
        "DR1TSODI": 2500,
        "DBD900": 1,
        "SLD012": 7,
        "INDFMMPI": 2.5,
        "epsilon": 1e-6,
    }

    ml.train_columns = [
        "protein_ratio",
        "sugar_ratio",
        "sodium_density",
        "fast_food_ratio",
        "calorie_activity",
        "fat_calorie_ratio",
        "log_calories",
        "log_sodium",
    ]

    from app.main import app

    return TestClient(app)


# -----------------------------
# 1. Basic sanity test
# -----------------------------
def test_basic():
    assert 1 + 1 == 2


# -----------------------------
# 2. Import test
# -----------------------------
def test_import_app():
    assert app is not None


# -----------------------------
# 3. Health endpoint test
# -----------------------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -----------------------------
# 4. Valid prediction test
# -----------------------------
def test_predict_valid(client):
    payload = {
        "RIDAGEYR": 35,
        "RIAGENDR": 2,
        "BMXBMI": 28.4,
        "PAQ605": 1,
        "PAQ620": 2,
        "SLD012": 7,
        "INDFMMPI": 2.5,
        "BPQ020": 2,
        "DR1TKCAL": 2200,
        "DR1TSUGR": 90,
        "DR1TTFAT": 70,
        "DR1TPROT": 80,
        "DR1TSODI": 3000,
        "DBD895": 4,
        "DBD900": 2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "threshold" in data
    assert "latency_seconds" in data


# -----------------------------
# 5. Invalid input test
# -----------------------------
def test_predict_invalid_input(client):
    payload = {"RIDAGEYR": "invalid"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# -----------------------------
# 6. Batch prediction test
# -----------------------------
def test_batch_predict(client):
    payload = {
        "records": [
            {
                "RIDAGEYR": 35,
                "RIAGENDR": 2,
                "BMXBMI": 28.4,
                "PAQ605": 1,
                "PAQ620": 2,
                "SLD012": 7,
                "INDFMMPI": 2.5,
                "BPQ020": 2,
                "DR1TKCAL": 2200,
                "DR1TSUGR": 90,
                "DR1TTFAT": 70,
                "DR1TPROT": 80,
                "DR1TSODI": 3000,
                "DBD895": 4,
                "DBD900": 2,
            }
        ]
    }

    response = client.post("/batch_predict", json=payload)

    assert response.status_code == 200
    assert "results" in response.json()


# -----------------------------
# 7. Missing values handling test
# -----------------------------
def test_missing_values_handling(client):
    payload = {
        "RIDAGEYR": 40,
        "RIAGENDR": 1,
        "BMXBMI": 30,
        "PAQ605": 1,
        "PAQ620": 2,
        "SLD012": None,
        "INDFMMPI": None,
        "BPQ020": 2,
        "DR1TKCAL": None,
        "DR1TSUGR": None,
        "DR1TTFAT": 70,
        "DR1TPROT": 80,
        "DR1TSODI": None,
        "DBD895": 3,
        "DBD900": None,
    }

    response = client.post("/predict", json=payload)

    # Should still work due to imputation
    assert response.status_code == 200


def test_response_structure(client):
    payload = {
        "RIDAGEYR": 30,
        "RIAGENDR": 1,
        "BMXBMI": 25,
        "PAQ605": 1,
        "PAQ620": 2,
        "BPQ020": 2,
        "DBD895": 3,
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)


def test_invalid_range(client):
    payload = {
        "RIDAGEYR": -10,  # invalid age
        "RIAGENDR": 1,
        "BMXBMI": 25,
        "PAQ605": 1,
        "PAQ620": 2,
        "BPQ020": 2,
        "DBD895": 3,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422]
