# Obesity Risk Prediction Platform

Production-ready machine learning platform that predicts obesity risk from lifestyle, dietary, and health indicators using XGBoost. The platform integrates automated feature engineering, SHAP-based explainability, LLM-generated prediction summaries, monitoring, Dockerized deployment, and GitHub Actions CI/CD, and is continuously deployed on AWS EC2.

## Model Performance

- **Recall**: 89%
- **Precision**: 46%
- **F1-Score**: 60%

## System Architecture

```

 Validation
      │
Feature Engineering
      │
XGBoost Inference
      │
SHAP Explainability
      │
LLM Explanation
      │
Monitoring & Logging
      │
JSON Response


```
## Production Architecture

                    GitHub
                       │
                 Push / Pull Request
                       │
             GitHub Actions CI/CD
        (Tests • Lint • Docker Build)
                       │
                 Docker Image
                       │
                Deploy to AWS EC2
                       │
                 FastAPI Application
                       │
      ┌────────────┬─────────────┬──────────────┐
      │            │             │              │
 Validation   Feature Eng.   XGBoost Model   Prediction
      │            │             │              │
      └────────────┴─────────────┘──────────────┘
                       │
          SHAP Explainability Engine
                       │
          Monitoring & Logging Layer
                       │
                JSON API Response
```

**Components:**
- **API Layer** ([app/main.py](app/main.py)): 3 endpoints for health, single, batch predictions
- **Model Loader** ([app/model_loader.py](app/model_loader.py)): Loads model artifacts once at startup for efficient inference
- **Preprocessor** ([app/preprocess.py](app/preprocess.py)): Missing value imputation + 12 engineered features
- **Validation** ([app/schema.py](app/schema.py)): Pydantic schemas for type safety

## Key Features

- FastAPI REST API for real-time predictions
- XGBoost classification model
- Automated feature engineering pipeline
- SHAP-based explainable AI
- LLM-generated human-readable prediction explanations
- Real-time prediction latency tracking
- Prediction monitoring endpoints
- Batch inference support
- Structured request validation using Pydantic
- Dockerized deployment
- GitHub Actions CI/CD
- Continuous deployment to AWS EC2

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

# Explainable AI

The platform integrates **SHAP (SHapley Additive Explanations)** to improve transparency and interpretability of every prediction.

### Explainability Features

- Global feature importance
- Local prediction explanations
- SHAP summary plots
- Waterfall plots
- Feature contribution analysis
- Human-readable explanations generated using an LLM

This enables both developers and end users to understand *why* the model produced a prediction instead of treating it as a black box.

---

# Monitoring

The platform continuously tracks inference behavior to improve reliability and detect potential issues.

### API Monitoring

- Request latency
- Average inference time
- Request count
- Error count
- Endpoint health

### Model Monitoring

- Prediction class distribution
- Feature distribution statistics
- Missing feature monitoring
- Schema validation failuress

Monitoring endpoints can be used to inspect model health during deployment.

---

# Logging

Structured JSON logging is implemented throughout the application to support request tracing, debugging, and production monitoring.

Each request records

- Request ID
- Timestamp
- Endpoint
- Prediction probability
- Prediction latency
- Error details 

# Deployment

The application is continuously deployed to an Amazon EC2 instance.

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
  "prediction_id": "dd443577-5c55-43ca-a193-2e2cafe4290a",
  "timestamp": "2026-07-06T20:19:08.082826+00:00",
  "prediction": 1,
  "probability": 0.80,

  "top_factors": [
    {
      "feature": "Protein Ratio",
      "value": 0.036,
      "impact": 0.50
    },
    {
      "feature": "Log Calories",
      "value": 7.69,
      "impact": 0.40
    },
    {
      "feature": "Calories From Fat",
      "value": 0.032,
      "impact": 0.30
    }
  ],

  "explanation": "The model predicts obesity with an 80% probability. The strongest contributors were Protein Ratio, Log Calories, and Calories From Fat.",

  "latency_seconds": 1.86
}
```

### `/batch_predict` (POST)
Multiple predictions: `{"records": [...]}`

See [Swagger UI](http://localhost:8000/docs) for full schema.

---

## `/monitor` (GET)

Returns monitoring statistics including

- prediction distribution
- average latency
- request count
- feature statistics

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

## CI Pipeline

- **GitHub Actions**: Runs on every push/PR
- **Linting**: Flake8, Black formatting
- **Testing**: Pytest with ≥75% coverage
- **Build**: Docker image validation

See `.github/workflows/ci.yml` for details.

## CD Pipeline

After the CI pipeline completes successfully, the application is automatically deployed to an Amazon EC2 instance.

## Deployment

### Production

The application is deployed on an Amazon EC2 instance using Docker Compose.

The deployed service exposes:

- `/docs` – Interactive Swagger UI
- `/health` – Health check endpoint
- `/predict` – Single prediction
- `/batch_predict` – Batch prediction
- `/monitor` – Monitoring metrics

### Local

```bash
uv sync
uvicorn app.main:app --reload
```
API Documentation
```bash
http://localhost:8000/docs

```
### Production

The application is hosted on an Amazon EC2 instance using Docker Compose.

Live API documentation is available on the deployed EC2 instance.

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
- **Explainability:** SHAP
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

**Status:** Production-Ready Prototype | **Version:** 1.1.0 | **Last Updated:** July 13, 2026
