# Lifestyle-Based Obesity Risk Prediction System
## 2-Week Sprint Plan 

**Status:** Foundation complete | Ready for optimization & deployment  
**Scope:** Features engineering, deployment, monitoring  
**Last Updated:** April 15, 2026

---

## 1. Problem Definition & Business Context

### What Are We Solving?
Build an ML system to **predict obesity risk** in individuals based on lifestyle factors (physical activity, sleep, socioeconomic status) from NHANES data. Enable healthcare providers to identify at-risk patients early for personalized interventions.

---

## 2. Two-Week Sprint Overview

This plan evolves as we learn. **Update weekly** when we discover changes.

### Week 1: Model Optimization & Artifacts (Apr 15-21)
- Optimize and validate model performance
- Engineered domain-driven features (diet ratios, activity interactions)
- Created versioned model artifacts (model, imputer, train_columns, threshold)
- Ensured preprocessing consistency between training and inference
- Structured repository with clean version control

### Week 2: Deployment & Operations (Apr 22-28)
- Build FastAPI service(/health, /predict, /batch_predict)
- Containerize with Docker
- Implement CI/CD pipeline
- Add monitoring & logging
- Complete documentation
- Release v1.0.0

---

## 3. Detailed Sprint Breakdown

### WEEK 1: Model Foundation & Deployment Readiness

#### Phase 1A: Model Optimization (completed) 
**Objective:** Validate & optimize logistic regression model

Activities:
- Performed feature analysis and domain-driven feature engineering
- Created engineered features:
    dietary ratios (protein, sugar, fat vs calories)
    sodium density
    calorie-activity interaction
    log transformations for skewed features
- Evaluated model using ROC, precision-recall tradeoffs
- Tuned classification threshold to 0.35 to improve recall

**Outcome:** 
- Achieved recall-focused model suitable for healthcare use case
- Established tradeoff between recall and precision 

---

#### Phase 1B: Model + Data Artifacts (completed)
**Objective:** Prepare reproducible, versioned model for deployment

Activities:
- Saved artifacts using joblib:
    model.joblib
    train_columns.joblib
    imputer.joblib
    threshold.joblib
- Ensured all preprocessing steps (missing values, feature engineering) are reproducible
- Removed duplicate logic and consolidated imputation into a single artifact

**Why this matters:** Guarantees consistent predictions across environments and prevents training–inference mismatch
**Deliverables:** Versioned artifacts in artifacts/model/  

---

#### Phase 1C: Feature Engineering (Completed)
**Objective:** Transform raw NHANES data into meaningful predictive features

Activities:
- Designed domain-informed features:
    protein_ratio, sugar_ratio, fat_calorie_ratio
    sodium_density
    diet_quality score
    calorie_activity interaction
- Added missing value indicators for key dietary variables
- Applied log transformations for skewed distributions

**Outcome:** Improved model’s ability to capture lifestyle patterns affecting obesity risk

---

#### Phase 1D: Git Setup & Versioning (Completed)
**Objective:** Establish version control and branching strategy

Activities:
- Implemented simplified branching strategy:
    - main → stable, production-ready code
    - dev → active development and feature work
- Cleaned repository:
    removed unnecessary files (__pycache__)
    added proper .gitignore
- Managed artifacts and dependencies (uv.lock, environment reproducibility)
 
**Deliverables:** Clean, structured GitHub repository

---

### WEEK 2: API, Deployment, Monitoring

#### Phase 2A: API Development (Completed)
**Objective:** Build production-ready prediction service

Activities:
- Built FastAPI application with endpoints:
    - /health
    - /predict (single inference)
    - /batch_predict (batch inference)
- Implemented Pydantic models for strict input validation
- Integrated preprocessing pipeline into API:
    - missing value handling via imputer
    - feature engineering
    - column alignment using train_columns
- Loaded all artifacts at startup:
    - model, imputer, threshold
- Implemented structured logging:
    - request logging
    - prediction logging
    - error logging
- Added global exception handling
- Implemented threshold-based classification (0.35)
- Added latency tracking per request

---

#### Phase 2B: Docker & Local Deployment 
**Objective:** Containerize the FastAPI-based ML prediction service to ensure consistent, reproducible environments across development and production.

Activities
- Dockerfile:
    - Built container using Python 3.10 slim, installed dependencies via pyproject.toml and uv.lock, and configured FastAPI with uvicorn.
- docker-compose.yml:
    - Set up service with port mapping, container name, and volume for artifacts.
- Local Testing:
    - Built and ran container using docker-compose; verified /docs, /health, /predict, and /batch_predict.
- Debugging:
    - Checked logs, validated model loading, and ensured preprocessing + validation worked inside container.
           
**Why this matters:** Deploy same code to dev/prod with zero environment surprises  
**Deliverables:** Dockerfile, docker-compose.yml, deployment guide  

---

#### Phase 2C: CI/CD Pipeline 
**Objective:** Automate code validation to ensure reliability and consistency before merging changes

### Continuous Integration (CI)

Activities:
- Configured GitHub Actions workflow triggered on push and pull requests
- Set up Python 3.12 environment using `actions/setup-python`
- Installed dependencies using `uv` for consistent and reproducible environments
- Implemented linting using `flake8` and formatting checks using `black`
- Developed unit tests using `pytest` for API endpoints and preprocessing logic
- Added test coverage reporting using `pytest-cov`
- Handled mocking of model artifacts to enable isolated test execution
- Validated Docker image build within CI to ensure container readiness
- Debugged CI issues including dependency mismatches, import paths, and environment inconsistencies

### Continuous Deployment (CD)

- Manually deployed Dockerized FastAPI application on AWS EC2
- Configured EC2 instance with Docker and required dependencies
- Built and ran containerized application exposing API endpoints
- Verified deployment using live API endpoints and Swagger UI
- Implemented container restart policy for basic reliability

**Why this matters:**  
- CI ensures code quality, catches bugs early, and validates functionality in a clean environment  
- CD enables reliable deployment and demonstrates the ability to move from development to production
**Deliverables:**  
- `.github/workflows/ci.yml`  
- Automated test suite with API and preprocessing coverage  
- Dockerized application with validated build process  
- Live deployment on AWS EC2  

---

#### Phase 2D: Monitoring & Logging  
**Objective:** Detect model drift and operational issues

Activities:
- API logging (requests, predictions, errors) → `logs/api.log`
- Track prediction latency, error rates
- Data drift detection: compare feature distributions vs training data
- Log alerts for drift thresholds
- Create `MONITORING_GUIDE.md`—when/how to retrain

**Why this matters:** Catch model degradation early; know when to retrain  
**Deliverables:** Monitoring system, drift detector, logging setup  

---

#### Phase 2E: Documentation 
**Objective:** Enable team onboarding and handoff

Activities:
- README: problem, quick start (local + Docker), examples
- API_DOCUMENTATION.md: all endpoints, request/response samples
- DEPLOYMENT_GUIDE.md: step-by-step deployment
- MODEL_CARD.md: model description, limitations, ethical considerations
- CONTRIBUTING.md: code style, PR process
- TROUBLESHOOTING.md: common issues & fixes

**Why this matters:** Reduces onboarding time; documents assumptions  
**Deliverables:** 6+ docs, comprehensive README  

---

#### Phase 2F: Integration Testing 
**Objective:** Verify end-to-end system works

Activities:
- Data → model → API → response integration tests
- Load testing: verify API responds in <200ms under load
- Final code review for quality & security
- Pre-deployment checklist

**Why this matters:** Catch integration bugs before production  
**Deliverables:** Test reports, performance benchmarks  

---

## 5. How This Plan Updates

**Triggers for change:**
- Model performance below targets → revisit feature engineering
- Data drift detected → trigger retraining discussion
- Infrastructure constraints → adjust deployment approach
- Stakeholder feedback → update metric targets or priorities

**Version tracking:** Use git to track plan changes (update CHANGELOG.md)

---

## 6. Success Metrics & Targets

### Model Performance (Apr 15 baseline)
| Metric | Target | Why |
|--------|--------|-----|
| **Recall** | ≥60% | Minimize false negatives—catch patients we should |
| **Precision** | ≥65% | Ensure flagged patients are actually at-risk; maintain provider trust |
| **Accuracy** | ≥70% | Overall system correctness |
| **F1-Score** | ≥0.62 | Balanced recall/precision harmony |

### Deployment Targets (Apr 22+)
| Component | Target | Success Criteria |
|-----------|--------|-----------------|
| **API Latency** | <200ms | Measured under typical load |
| **Test Coverage** | ≥75% | Unit + integration tests |
| **Uptime** | >99% | After day 1 deployment |
| **Documentation** | 100% | README, API docs, deployment guide complete |

---

## 7. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Model performance below targets | Blocks deployment | Revisit features/hyperparams by Apr 17 |
| API bottleneck | Service degradation | Load test early (Apr 27) |
| Missing dependencies | Build failures | Maintain updated requirements.txt |
| Data drift | Model decay | Implement drift detector (Apr 25) |
| Team bandwidth | Schedule slippage | Weekly retrospective to reprioritize |

---

## 9. Repository Structure

---

## 14. Quick Links & References

- **NHANES Data Portal:** https://wwwn.cdc.gov/nchs/nhanes/

---


