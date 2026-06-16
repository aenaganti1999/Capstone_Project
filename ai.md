# AI Collaboration Memory — Obesity Risk Prediction API

## Project Overview

This project is a production-style machine learning API for predicting obesity risk using healthcare and lifestyle-related features.

The primary objective is early risk detection rather than perfect precision. In healthcare screening systems, missing at-risk individuals (false negatives) is significantly more costly than over-flagging users.

The system is designed with:
- FastAPI-based inference
- strict schema validation
- train/inference consistency
- artifact-driven preprocessing
- low-latency predictions
- CI/CD integration
- containerized deployment

---

# Core Business Objective

The goal is NOT just prediction accuracy.

The goal is:
- identify at-risk individuals early
- prioritize recall
- minimize false negatives
- provide production-safe inference

Healthcare reasoning:
Missing an at-risk patient is more dangerous than over-flagging a healthy patient.

Therefore:
Recall > Precision

Target Metrics:
- Recall ≈ 89%
- Precision ≈ 48%
- F1 ≈ 0.61

---

# Current Tech Stack

## Backend
- FastAPI
- Uvicorn
- Pydantic v2

## Machine Learning
- XGBoost
- Scikit-learn
- Pandas
- NumPy

## DevOps
- Docker
- GitHub Actions
- uv package manager

## Planned Cloud Deployment
- AWS EC2
- AWS S3
- AWS ECR

---

# IMPORTANT PROJECT RULES

## DO NOT Suggest

- requirements.txt
- Flask
- SQLite
- dynamic preprocessing at inference
- manual feature ordering
- loading model artifacts per request

This project intentionally uses:
- uv
- artifact-driven preprocessing
- strict schema validation
- startup-time artifact loading

---

# API Architecture

## Endpoints

### /health
Health check endpoint.

Purpose:
Verify API availability.

---

### /predict
Single prediction endpoint.

Input:
Single validated payload.

Output:
Prediction result + probability + latency.

---

### /batch_predict
Batch prediction endpoint.

Input:
List of records.

Output:
List of predictions.

Important:
Batch predictions MUST match single prediction behavior exactly.

---

# Input Schema Design

The API uses strict Pydantic validation.

Important fields:
- RIDAGEYR
- RIAGENDR
- BMXBMI
- ALQ111
- PAQ605
- PAQ620
- SLD012
- INDFMMPI
- BPQ020
- DR1TKCAL
- DR1TSUGR
- DR1TTFAT
- DR1TPROT
- DR1TSODI
- DBD895
- DBD900

---

# Validation Rules

## Gender Encoding

Accepted:
- 1 = Male
- 2 = Female

Invalid:
- negative values
- strings
- unknown categories

---

## Probability Constraints

Output probability:
0 ≤ probability ≤ 1

Threshold:
0 ≤ threshold ≤ 1

---

# Data Cleaning Decisions

## Sentinel Value Handling

Problem:
Dataset contained invalid placeholder values:
5.397605e-79

Fix:
Replace with NaN before preprocessing.

Reason:
These values distorted distributions and feature engineering.

---

# Missing Value Strategy

## Median Imputation Used For

- DR1TKCAL
- DR1TSUGR
- DR1TTFAT
- DR1TPROT
- DR1TSODI

Reason:
These are continuous nutritional features.

Median was chosen because:
- robust to outliers
- more stable for healthcare data

---

# Missing Indicators

Created:
- DR1TKCAL_missing
- DR1TSUGR_missing
- DR1TTFAT_missing
- DR1TPROT_missing
- DR1TSODI_missing

Reason:
Missingness itself may contain predictive signal.

Status:
Kept in final pipeline.

---

# Feature Engineering Decisions

## protein_ratio

Formula:
DR1TPROT / DR1TKCAL

Reason:
Absolute protein intake was misleading because higher calorie intake naturally increases protein intake.

Ratio normalized protein quality relative to calories.

Result:
Improved class separation.

Status:
Kept.

---

## sugar_ratio

Formula:
DR1TSUGR / DR1TKCAL

Reason:
Sugar density is more meaningful than total sugar intake.

Result:
Improved recall for high-risk individuals.

Status:
Kept.

---

## sodium_density

Formula:
DR1TSODI / DR1TKCAL

Reason:
Relative sodium consumption was more predictive than raw sodium intake.

Status:
Kept.

---

## calorie_activity

Formula:
DR1TKCAL * PAQ605

Reason:
Wanted interaction between calorie intake and physical activity.

Result:
Captured sedentary high-calorie lifestyle patterns.

Status:
Kept.

---

## fat_calorie_ratio

Formula:
DR1TTFAT / DR1TKCAL

Reason:
Relative fat composition is more informative than total fat.

Status:
Kept.

---

## diet_quality

Formula:
protein_ratio - sugar_ratio

Reason:
Wanted a simplified dietary balance indicator.

Result:
Moderate improvement.

Status:
Kept.

---

## log_calories

Formula:
log1p(DR1TKCAL)

Reason:
Calorie distribution was heavily skewed.

Result:
Improved model stability.

Status:
Kept.

---

## log_sodium

Formula:
log1p(DR1TSODI)

Reason:
Sodium distribution contained extreme outliers.

Status:
Kept.

---

# Model Experimentation

## Logistic Regression

### Why Tried
Wanted:
- interpretability
- fast inference
- baseline comparison

### What Failed
- underfit nonlinear interactions
- poor minority class recall

### Metrics
- Recall ≈ 61%
- F1 ≈ 0.42

### Decision
Rejected.

---

## Random Forest

### Why Tried
Wanted:
- nonlinear modeling
- ensemble robustness

### What Failed
- larger model size
- slower inference
- unstable probability calibration

### Metrics
- Recall ≈ 78%
- F1 ≈ 0.56

### Decision
Not selected.

---

## XGBoost (Final Model)

### Why Selected
Best balance of:
- recall
- nonlinear learning
- inference speed
- feature interaction modeling

### What Worked
- strong recall
- stable predictions
- fast inference
- robust tabular learning

### Final Metrics
- Recall ≈ 89%
- Precision ≈ 48%
- F1 ≈ 0.61

### Final Decision
Production model.

---

# Class Imbalance Strategy

## Problem

Positive class was underrepresented.

This caused:
- poor recall
- biased predictions toward majority class

---

## Solution

Used:
scale_pos_weight

Reason:
Improve minority class sensitivity.

---

# Threshold Tuning Strategy

## Why Threshold Tuning Was Important

Default threshold (0.5) did not maximize healthcare objectives.

Goal:
Increase recall while maintaining acceptable precision.

---

## Tested Thresholds

- 0.50
- 0.55
- 0.60
- 0.65

---

## Final Decision

Selected threshold prioritizing recall.

Reason:
Healthcare systems prefer over-flagging over missing high-risk patients.

---

# Artifact-Driven Preprocessing

## Important Design Decision

Artifacts are persisted and reused during inference.

Artifacts:
- model.joblib
- imputer.joblib
- train_columns.joblib

---

## Why This Matters

Prevents:
- train/inference mismatch
- column ordering issues
- inconsistent preprocessing

---

# VERY IMPORTANT RULE

DO NOT dynamically regenerate feature columns during inference.

Always use:
train_columns.joblib

Reason:
Dynamic column generation previously caused prediction inconsistencies.

---

# Testing Strategy

## Goal

Ensure:
- schema safety
- prediction consistency
- stable inference
- production-safe APIs

---

# Test Categories

## 1. Schema Validation Tests

Purpose:
Reject malformed payloads safely.

Cases:
- negative age
- invalid gender
- missing fields
- invalid numeric ranges

Expected:
HTTP 422 validation errors.

---

## 2. Batch vs Single Prediction Consistency

Test:
test_batch_vs_single_consistency()

Purpose:
Ensure preprocessing is identical.

Reason:
Batch and single predictions must behave identically.

---

## 3. Threshold Behavior Tests

Purpose:
Validate classification logic.

Cases:
- probability below threshold
- probability above threshold

Reason:
Threshold directly impacts recall-sensitive healthcare predictions.

---

## 4. Model Failure Tests

Purpose:
Ensure graceful API failure handling.

Method:
Mock:
predict_proba()

Expected:
HTTP 500 with structured error message.

---

# Known Issues Encountered

## Problem 1 — Feature Ordering Mismatch

Cause:
Dynamic feature generation during inference.

Impact:
Wrong predictions.

Fix:
Persisted:
train_columns.joblib

Lesson:
Never infer feature ordering dynamically.

---

## Problem 2 — High Latency

Cause:
Artifacts loaded per request.

Impact:
800ms+ latency.

Fix:
Load artifacts globally during startup.

Result:
~70ms latency.

---

## Problem 3 — Invalid Schema Inputs

Cause:
Unvalidated payloads.

Impact:
Silent failures.

Fix:
Strict Pydantic validation.

---

# Docker Decisions

## Why Docker Used

Goal:
Environment consistency.

Benefits:
- reproducible deployments
- dependency isolation
- consistent inference behavior

---

# CI/CD Decisions

## CI Pipeline Includes

- uv sync
- flake8
- black
- pytest
- coverage checks
- Docker image build

---

# Deployment Architecture (Planned)

## Final Deployment Flow

Docker Image → ECR
Artifacts → S3
EC2 → Pulls image from ECR

Reason:
Production-style ML deployment architecture.

---

# AI Assistant Instructions

## When Suggesting Code

Prioritize:
- production-safe patterns
- explicit validation
- deterministic preprocessing
- modular design

Avoid:
- toy implementations
- hidden preprocessing
- magic feature generation
- implicit schema assumptions

---

# Debugging Priorities

When debugging predictions:

Check in this order:
1. feature ordering
2. artifact loading
3. preprocessing mismatch
4. schema validation
5. threshold logic

---

# Lessons Learned

## Lesson 1

Inference consistency is more important than experimentation flexibility.

---

## Lesson 2

Feature engineering quality mattered more than trying many models.

---

## Lesson 3

Strict schema validation prevented major production failures.

---

## Lesson 4

Healthcare ML systems should optimize business impact, not just accuracy.

---

# Future Improvements

- MLflow integration
- Prometheus monitoring
- model drift detection
- feature store integration
- automated retraining pipeline
- ECS/Kubernetes deployment
- shadow deployment testing
- model explainability integration

---