import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import mlflow
import mlflow.sklearn

# Load data
data = pd.read_csv("../data/simulated_claims.csv")

# Feature engineering example
data['days_since_last_claim'] = (pd.Timestamp('2023-07-15') - pd.to_datetime(data['last_claim_date'])).dt.days
features = ['amount', 'transaction_count', 'total_claims', 'days_since_last_claim']
X = data[features]
y = data['fraud_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Start MLflow experiment
mlflow.set_experiment("Fraudulent_Claim_Detection")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    print(f"AUC-ROC: {auc:.4f}")

    mlflow.log_metric("AUC-ROC", auc)

    # Save model
    mlflow.sklearn.log_model(model, "random_forest_model")
    joblib.dump(model, "../models/random_forest_model.pkl")

    # Classification report
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
