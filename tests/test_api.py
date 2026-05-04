from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch
import pytest


@pytest.fixture
def client():
    from unittest.mock import MagicMock
    import app.model_loader as ml

    # Mock model
    ml.model = MagicMock()
    ml.threshold = 0.5

    ml.model.predict.return_value = [1]
    ml.model.predict_proba.return_value = np.array([[0.2, 0.8]])

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


# 1. Health endpoint test
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# 2. Valid prediction test
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


# 3. Invalid input test
def test_predict_invalid_input(client):
    payload = {"RIDAGEYR": "invalid"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# 4. Batch prediction test
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
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 1
    assert "prediction" in data["results"][0]


# 5. Missing values handling test
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


def test_threshold_behavior(client):
    import app.model_loader as ml
    import numpy as np

    # Force probability below threshold
    ml.model.predict_proba.return_value = np.array([[0.8, 0.2]])
    ml.threshold = 0.5

    payload = {
        "RIDAGEYR": 35,
        "RIAGENDR": 1,
        "BMXBMI": 27.5,
        "PAQ605": 2,
        "PAQ620": 3,
        "SLD012": 7,
        "INDFMMPI": 2.5,
        "BPQ020": 1,
        "DR1TKCAL": 2200,
        "DR1TSUGR": 60,
        "DR1TTFAT": 70,
        "DR1TPROT": 80,
        "DR1TSODI": 2500,
        "DBD895": 4,
        "DBD900": 2,
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert data["prediction"] == 0


def test_batch_vs_single_consistency(client):
    payload = {
        "RIDAGEYR": 35,
        "RIAGENDR": 1,
        "BMXBMI": 27.5,
        "PAQ605": 2,
        "PAQ620": 3,
        "SLD012": 7,
        "INDFMMPI": 2.5,
        "BPQ020": 1,
        "DR1TKCAL": 2200,
        "DR1TSUGR": 60,
        "DR1TTFAT": 70,
        "DR1TPROT": 80,
        "DR1TSODI": 2500,
        "DBD895": 4,
        "DBD900": 2,
    }

    single = client.post("/predict", json=payload).json()

    batch = client.post("/batch_predict", json={"records": [payload]}).json()

    assert single["prediction"] == batch["results"][0]["prediction"]


def test_model_failure(client):

    with patch(
        "app.model_loader.model.predict_proba", side_effect=Exception("Model crashed")
    ):

        payload = {
            "RIDAGEYR": 35,
            "RIAGENDR": 1,
            "BMXBMI": 27.5,
            "PAQ605": 2,
            "PAQ620": 3,
            "SLD012": 7,
            "INDFMMPI": 2.5,
            "BPQ020": 1,
            "DR1TKCAL": 2200,
            "DR1TSUGR": 60,
            "DR1TTFAT": 70,
            "DR1TPROT": 80,
            "DR1TSODI": 2500,
            "DBD895": 4,
            "DBD900": 2,
        }

        response = client.post("/predict", json=payload)

        assert response.status_code == 500


def test_invalid_gender(client):
    payload = {
        "RIDAGEYR": 45,
        "RIAGENDR": -1,  # invalid
        "BMXBMI": 28.5,
        "PAQ620": 3.0,
        "SLD012": 7.0,
        "PAQ605": 1,
        "INDFMMPI": 2.5,
        "BPQ020": 1.0,
        "DR1TKCAL": 2000,
        "DR1TSUGR": 50,
        "DR1TTFAT": 65,
        "DR1TPROT": 75,
        "DR1TSODI": 2300,
        "DBD895": 15,
        "DBD900": 2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
