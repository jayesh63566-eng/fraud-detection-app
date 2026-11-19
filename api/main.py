"""FastAPI service for the fraud model.  Run:  uvicorn api.main:app --reload --port 8000"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.deploy import bundle_from_bytes, predict, red_flags

MODEL_PATH = os.environ.get("MODEL_PATH", "models/fraud_model.joblib")
app = FastAPI(title="Fraud Detection API", version="1.0")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, f"Model file nahi mili: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            _bundle = bundle_from_bytes(f.read())
    return _bundle


class Transaction(BaseModel):
    step: int = 1
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    nameOrig: str = "C0"
    nameDest: str = "C0"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}


@app.get("/model-info")
def info():
    b = bundle()
    return {"name": b["name"], "version": b["version"], "created": b["created"],
            "threshold": b["threshold"], "features": b["features"], "metrics": b["metrics"]}


@app.post("/predict")
def predict_one(t: Transaction):
    row = t.model_dump()
    out = predict(bundle(), pd.DataFrame([row])).iloc[0]
    return {"fraud_probability": float(out["fraud_probability"]), "decision": out["decision"],
            "risk": out["risk"], "reasons": red_flags(row)}


@app.post("/predict_batch")
def predict_batch(items: list[Transaction]):
    df = pd.DataFrame([i.model_dump() for i in items])
    return predict(bundle(), df)[["fraud_probability", "decision", "risk"]].to_dict(orient="records")
