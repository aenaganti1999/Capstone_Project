from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch
import pytest


@pytest.fixture
def client():
    from unittest.mock import MagicMock
    import app.model_loader as ml

    ml.model = MagicMock()
    ml.threshold = 0.5

    ml.model.predict.return_value = [1]
    ml.model.predict_proba.return_value = np.array([[0.2, 0.8]])

    ml.explainer = MagicMock()
    ml.explainer.shap_values.return_value = np.array(
        [[0.5, 0.2, -0.1, 0.05, 0.01, 0.3, 0.4, 0.1]]
    )

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


@pytest.fixture
def sample_payload():
    return {
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
def test_missing_values_handling(client, sample_payload):

    response = client.post("/predict", json=sample_payload)

    # Should still work due to imputation
    assert response.status_code == 200


def test_threshold_behavior(client, sample_payload):
    import app.model_loader as ml
    import numpy as np

    # Force probability below threshold
    ml.model.predict_proba.return_value = np.array([[0.8, 0.2]])
    ml.threshold = 0.5

    response = client.post("/predict", json=sample_payload)
    data = response.json()

    assert data["prediction"] == 0


def test_batch_vs_single_consistency(client, sample_payload):

    single = client.post("/predict", json=sample_payload).json()

    batch = client.post("/batch_predict", json={"records": [sample_payload]}).json()

    assert single["prediction"] == batch["results"][0]["prediction"]


def test_model_failure(client, sample_payload):

    with patch(
        "app.model_loader.model.predict_proba", side_effect=Exception("Model crashed")
    ):

        response = client.post("/predict", json=sample_payload)

        assert response.status_code == 500


def test_invalid_gender(client, sample_payload):
    sample_payload["RIAGENDR"] = -1

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 422


def test_shap_factors_present(client, sample_payload):

    response = client.post("/predict", json=sample_payload)

    data = response.json()

    assert "top_factors" in data
    assert len(data["top_factors"]) > 0


def test_shap_factor_schema(client, sample_payload):

    response = client.post("/predict", json=sample_payload)

    factor = response.json()["top_factors"][0]

    assert "feature" in factor
    assert "value" in factor
    assert "impact" in factor


def test_explanation_generated(client, sample_payload):
    response = client.post("/predict", json=sample_payload)

    assert "explanation" in response.json()


def test_explanation_not_empty(client, sample_payload):
    response = client.post("/predict", json=sample_payload)

    explanation = response.json()["explanation"]

    assert len(explanation) > 20


def test_monitoring_endpoint(client):
    response = client.get("/monitoring")

    assert response.status_code == 200


def test_monitoring_contains_latency(client):
    response = client.get("/monitoring")

    assert "average_latency_seconds" in response.json()


def test_monitoring_contains_distribution(client):
    response = client.get("/monitoring")

    assert "prediction_distribution" in response.json()


def test_drift_endpoint(client):
    response = client.get("/monitoring/drift")

    assert response.status_code == 200


def test_drift_summary_exists(client):
    response = client.get("/monitoring/drift")

    assert "summary" in response.json()


def test_prediction_under_5_seconds(client, sample_payload):
    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    data = response.json()

    assert data["latency_seconds"] < 5
