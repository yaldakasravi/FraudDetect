# Fraudulent Claim Detection System

**Keywords:** Classification, Imbalanced Data, Random Forest, Precision/Recall, SMOTE

---

## Project Overview

This project implements a robust machine learning pipeline for detecting fraudulent insurance claims using transaction-level data. The goal is to improve claim quality and reduce the insurer’s risk exposure by accurately identifying suspicious claims before payout.

---

## Key Features

- **Binary Classification Pipeline:** Developed models to classify claims as fraudulent or legitimate.
- **Handling Imbalanced Data:** Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the heavily skewed dataset.
- **Ensemble Modeling:** Utilized Random Forest and CatBoost classifiers to improve predictive performance and robustness.
- **Evaluation Metrics:** Achieved an AUC-ROC score of 0.84, complemented by precision-recall curves to balance false positives and false negatives.
- **Deployment:** Containerized models with Docker to enable scalable microservice architecture.
- **Monitoring:** Integrated custom MLflow dashboards for tracking prediction quality and model drift over time.

---

