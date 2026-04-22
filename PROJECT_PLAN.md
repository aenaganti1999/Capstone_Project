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
**Objective:** Containerize API for consistent environments

Activities:
- Create Dockerfile (Python 3.10 slim base)
- Create docker-compose.yml for local orchestration
- Test locally—verify API endpoints work in container
- Document deployment steps

**Why this matters:** Deploy same code to dev/prod with zero environment surprises  
**Deliverables:** Dockerfile, docker-compose.yml, deployment guide  

---

#### Phase 2C: CI/CD Pipeline 
**Objective:** Automate testing and deployment

Activities:
- GitHub Actions workflow: trigger on push/PR
- Linting (pylint/flake8) & formatting (black)
- Unit tests (pytest, ≥75% coverage)
- Docker image build on merge to `main`
- (Optional) Deploy to cloud (AWS/GCP)

**Why this matters:** Catch bugs early, reduce manual errors, faster releases  
**Deliverables:** `.github/workflows/ci.yml`, test suite  

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

#### Phase 2G: Release & Presentation (Apr 28) 
**Objective:** Ship v1.0.0 and present to stakeholders

Activities:
- Merge `dev` → `main`, create GitHub Release
- Tag as `v1.0.0`, update CHANGELOG.md
- Prepare 5-min demo + slides
- Live API demo (predictions on real data)

**Why this matters:** Formal handoff, stakeholder communication  
**Deliverables:** v1.0.0 release, presentation  

---

## 4. What We're Doing First vs. Next (Priority Order)

### Must Complete (Critical Path) 
1. **Model optimization & testing** 
2. **API endpoints**  
3. **Docker containerization** 
4. **Basic CI/CD** 
5. **Documentation** 

### Should Complete (High Value) 
6. Model artifacts versioning 
7. Git setup & branching 
8. Monitoring & logging 
9. Integration testing 

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

## 8. Artifacts & Deliverables Summary

### Week 1 Deliverables
- `models/v1/logistic_regression_model.pkl`, `scaler.pkl`, feature metadata
- Data splits and quality report
- `docs/FEATURE_ENGINEERING.md`, `GIT_WORKFLOW.md`
- Git repo with `main`, `dev`, `production` branches

### Week 2 Deliverables  
- `api/main.py` - FastAPI service with 3 endpoints
- `Dockerfile`, `docker-compose.yml`
- `.github/workflows/ci.yml` - GitHub Actions CI/CD
- `monitoring/drift_detector.py`, logging setup
- Comprehensive docs: README, API_DOCUMENTATION, DEPLOYMENT_GUIDE, MODEL_CARD
- Test suite with ≥75% coverage
- `v1.0.0` GitHub Release

---

## 9. Repository Structure
```
NHANES_Capstone/
├── analyzedata.ipynb                      # Main model notebook (Week 0)
├── PROJECT_PLAN.md                        # This file—update weekly
├── README.md                              # (Completed Apr 26)
│
├── models/                                # Model versioning
│   ├── v0/                                # (Archive)
│   └── v1/
│       ├── logistic_regression_model.pkl
│       ├── scaler.pkl
│       ├── feature_names.json
│       └── model_metadata.json
│
├── data/
│   ├── raw/                               # Original NHANES
│   ├── processed/                         # Cleaned data
│   └── splits/                            # Train/test files
│
├── api/                                   # FastAPI service (Week 2)
│   ├── main.py
│   ├── models.py
│   ├── config.py
│   └── requirements.txt
│
├── tests/                                 # Automated tests
│   ├── test_api_basic.py
│   ├── test_end_to_end.py
│   └── test_load.py
│
├── monitoring/                            # Logging & monitoring (Week 2)
│   ├── drift_detector.py
│   └── alert_system.py
│
├── docs/                                  # Documentation
│   ├── FEATURE_ENGINEERING.md
│   ├── API_DOCUMENTATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   ├── MODEL_CARD.md
│   ├── MONITORING_GUIDE.md
│   ├── GIT_WORKFLOW.md
│   └── TROUBLESHOOTING.md
│
├── Dockerfile                             # Container (Apr 23)
├── docker-compose.yml                     # Compose config (Apr 23)
├── .github/workflows/ci.yml               # CI/CD (Apr 24)
│
├── .gitignore
├── VERSION.md
└── CHANGELOG.md
```

---

## 14. Quick Links & References

- **NHANES Data Portal:** https://wwwn.cdc.gov/nchs/nhanes/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **GitHub Actions:** https://docs.github.com/en/actions
- **Docker Docs:** https://docs.docker.com/

---

## Questions? Check These Docs First

| Question | Doc |
|----------|-----|
| How do we encode features? | `FEATURE_ENGINEERING.md` |
| How do I deploy the API? | `DEPLOYMENT_GUIDE.md` |
| What are the model limits? | `MODEL_CARD.md` |
| When should we retrain? | `MONITORING_GUIDE.md` |
| How do we use git? | `GIT_WORKFLOW.md` |
| API not working? | `TROUBLESHOOTING.md` |


