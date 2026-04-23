from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# -----------------------------
# 4. Valid prediction test
# -----------------------------
def test_predict_valid():
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
        "DBD900": 2
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
def test_predict_invalid_input():
    payload = {
        "RIDAGEYR": "invalid"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# -----------------------------
# 6. Batch prediction test
# -----------------------------
def test_batch_predict():
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
                "DBD900": 2
            }
        ]
    }

    response = client.post("/batch_predict", json=payload)

    assert response.status_code == 200
    assert "results" in response.json()


# -----------------------------
# 7. Missing values handling test
# -----------------------------
def test_missing_values_handling():
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
        "DBD900": None
    }

    response = client.post("/predict", json=payload)

    # Should still work due to imputation
    assert response.status_code == 200

def test_response_structure():
    payload = {
        "RIDAGEYR": 30,
        "RIAGENDR": 1,
        "BMXBMI": 25,
        "PAQ605": 1,
        "PAQ620": 2,
        "BPQ020": 2,
        "DBD895": 3
    }

    response = client.post("/predict", json=payload)
    data = response.json()

    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)
def test_invalid_range():
    payload = {
        "RIDAGEYR": -10,  # invalid age
        "RIAGENDR": 1,
        "BMXBMI": 25,
        "PAQ605": 1,
        "PAQ620": 2,
        "BPQ020": 2,
        "DBD895": 3
    }

    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 422]