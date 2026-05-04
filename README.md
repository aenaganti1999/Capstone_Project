# Obesity Risk Prediction API

**FastAPI ML service** predicting obesity risk from lifestyle factors (diet, activity, sleep). Built with Python, XGBoost, and deployed on AWS EC2 and accessible via a live API endpoint

## Model Performance

- **Recall**: 89%
- **Precision**: 46%
- **F1-Score**: 60%

## System Architecture

```
Client → FastAPI → Validation → Preprocessing (feature engineering) → Model → Response
```

**Components:**
- **API Layer** ([app/main.py](app/main.py)): 3 endpoints for health, single, batch predictions
- **Model Loader** ([app/model_loader.py](app/model_loader.py)): Loads model artifacts once at startup for efficient inference
- **Preprocessor** ([app/preprocess.py](app/preprocess.py)): Missing value imputation + 12 engineered features
- **Validation** ([app/schema.py](app/schema.py)): Pydantic schemas for type safety

## Key Features

- FastAPI-based ML inference API
- Real-time predictions with latency tracking
- Batch prediction support
- CI pipeline with GitHub Actions
- Dockerized deployment
- Hosted on AWS EC2

## Quick Start

```bash
# Local
git clone https://github.com/yourusername/capstone_project.git
cd Capstone_Project
source .venv/bin/activate
pip install -r pyproject.toml
uvicorn app.main:app --reload

# Docker
docker-compose up --build

# Access: 
http://localhost:8000/docs
```

## Prerequisites

- Python 3.11+ or Docker
- 8GB RAM
- Port 8000 available

---

## API Endpoints

### `/health` (GET)
Health check endpoint.
```json
{ "status": "ok" }
```

### `/predict` (POST)
Single prediction from 16 lifestyle features.

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RIDAGEYR": 45, 
    "RIAGENDR": 1, 
    "BMXBMI": 28.5,
    "PAQ605": 2.0, 
    "PAQ620": 3.0, 
    "SLD012": 7.0,
    "INDFMMPI": 2.5, 
    "BPQ020": 1.0,
    "DR1TKCAL": 2000, 
    "DR1TSUGR": 50, 
    "DR1TTFAT": 65,
    "DR1TPROT": 75, 
    "DR1TSODI": 2300,
    "DBD895": 15, 
    "DBD900": 2
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.75,
  "threshold": 0.35,
  "latency_seconds": 0.052
}
```

Fields: `prediction` (0/1), `probability` (0-1), `threshold`,`latency_seconds`

### `/batch_predict` (POST)
Multiple predictions: `{"records": [...]}`

See [Swagger UI](http://localhost:8000/docs) for full schema.

---

## Model Overview

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost + Feature Engineering |
| Training Data | NHANES (10K individuals) |
| Features | 16 input → 40+ engineered |
| Recall | 89% |
| Precision | 46% |
| F1-Score | 60% |

**Feature engineering:** Dietary ratios, activity metrics, log transformations, missing indicators.

---

## Project Structure

```
app/
  ├── main.py           # FastAPI endpoints
  ├── model_loader.py   # Artifact loading
  ├── preprocess.py     # Feature engineering
  └── schema.py         # Pydantic validation
artifacts/
  ├── model/            # model.joblib, threshold, etc.
  └── data/             # Training/test splits
tests/
  └── test_api.py       # Unit tests (pytest)
```

---

## Testing

```bash
# Unit tests
pip install pytest pytest-cov httpx
pytest --cov=app tests/

# Interactive testing
Open http://localhost:8000/docs (Swagger UI)
```

---

## Deployment

**Local:**
```bash
uvicorn app.main:app --reload
```

**Docker:**
```bash
docker build -t obesity-api:1.0 .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts obesity-api:1.0
```

**Production (AWS EC2):**
```bash
ssh -i key.pem ubuntu@ec2-ip
git clone repo && cd repo
docker-compose up -d
```

Access at: `http://your-ec2-ip:8000/docs`

---

## CI/CD Pipeline

- **GitHub Actions**: Runs on every push/PR
- **Linting**: Flake8, Black formatting
- **Testing**: Pytest with ≥75% coverage
- **Build**: Docker image validation

See `.github/workflows/ci.yml` for details.

---

## Performance

- **Single prediction:** <200ms (typical: 50-100ms)
- **Batch processing:** Linear scaling
- **Uptime:** 99%+ (auto-restart via Docker)

---

## Tech Stack

- **Language:** Python 3.11+
- **Framework:** FastAPI, Uvicorn
- **ML:** XGBoost, Scikit-learn, Pandas, NumPy
- **Validation:** Pydantic v2
- **Testing:** Pytest, HTTPx
- **Containerization:** Docker, Docker Compose
- **Deployment:** AWS EC2
- **Package Manager:** UV

---

## Key References

- [PROJECT_PLAN.md](PROJECT_PLAN.md) - Sprint timeline & milestones
- [Model Training](train.ipynb) - Training notebook
- [NHANES Data](https://wwwn.cdc.gov/nchs/nhanes/) - Datasets

---

**Status:** Production-Ready Prototype | **Version:** 1.1.0 | **Last Updated:** April 28, 2026
