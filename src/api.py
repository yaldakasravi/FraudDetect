from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.inference import predict_fraud
from src.utils import preprocess_input

app = FastAPI()

class ClaimData(BaseModel):
    amount: float
    transaction_count: int
    total_claims: int
    last_claim_date: str  # expect 'YYYY-MM-DD'

@app.post("/predict")
def predict(data: ClaimData):
    df = preprocess_input(data.dict())
    preds = predict_fraud(df)
    return {"fraud_probability": float(preds[0])}
