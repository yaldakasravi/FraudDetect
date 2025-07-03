import joblib
import pandas as pd

model = joblib.load("../models/random_forest_model.pkl")

def predict_fraud(input_df: pd.DataFrame):
    preds_proba = model.predict_proba(input_df)[:, 1]
    return preds_proba
