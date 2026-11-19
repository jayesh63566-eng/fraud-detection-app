"""Deployment helpers: model bundle, live prediction, red-flag explanations, API/Docker templates.
NOTE: must not import streamlit (the FastAPI service imports this module too)."""
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from core.features import engineer

RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
               "nameDest", "oldbalanceDest", "newbalanceDest"]


def build_bundle(name, pipeline, features, opts, threshold, metrics, train_cfg=None):
    return dict(name=name, pipeline=pipeline, features=list(features), opts=dict(opts), threshold=float(threshold),
                metrics={k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                         for k, v in metrics.items()},
                train_cfg=train_cfg or {}, created=datetime.now().isoformat(timespec="seconds"), version="1.0")


def bundle_to_bytes(bundle) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


def bundle_from_bytes(raw: bytes):
    return joblib.load(io.BytesIO(raw))


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the columns a user may not have (ids / step) so engineer() always works."""
    d = df.copy()
    if "step" not in d:
        d["step"] = 1
    if "nameOrig" not in d:
        d["nameOrig"] = "C0"
    if "nameDest" not in d:
        d["nameDest"] = "C0"
    return d


def predict(bundle, raw_df: pd.DataFrame) -> pd.DataFrame:
    raw = prepare_raw(raw_df)
    feat, _ = engineer(raw, bundle["opts"])
    X = feat.reindex(columns=bundle["features"]).fillna(0)
    prob = bundle["pipeline"].predict_proba(X)[:, 1]
    thr = bundle["threshold"]
    out = raw_df.copy()
    out["fraud_probability"] = np.round(prob, 4)
    out["decision"] = np.where(prob >= thr, "FRAUD", "OK")
    out["risk"] = [risk_band(p, thr) for p in prob]
    return out


def risk_band(p: float, thr: float) -> str:
    if p >= thr:
        return "HIGH"
    return "MEDIUM" if p >= thr / 2 else "LOW"


def red_flags(row: dict) -> list[str]:
    """Plain-language reasons an analyst can read (rule-based, independent of the model)."""
    f = []
    amt = float(row.get("amount", 0) or 0)
    oo, no = float(row.get("oldbalanceOrg", 0) or 0), float(row.get("newbalanceOrig", 0) or 0)
    od, nd = float(row.get("oldbalanceDest", 0) or 0), float(row.get("newbalanceDest", 0) or 0)
    t = str(row.get("type", ""))
    if t in ("TRANSFER", "CASH_OUT"):
        f.append(f"Transaction type {t}: fraud isi type mein hota hai.")
    if no == 0 and oo > 0:
        f.append("Sender ka account poora khaali ho gaya.")
    if abs(amt - oo) < 0.01 and amt > 0:
        f.append("Poora balance ek hi transaction mein bheja gaya.")
    if od == nd and amt > 0 and t in ("TRANSFER", "CASH_OUT"):
        f.append("Receiver ka balance nahi badla (paisa gayab/hidden account).")
    if abs(no + amt - oo) > 1 and t != "CASH_IN":
        f.append("Sender balance ka hisaab match nahi karta (errorBalanceOrig ≠ 0).")
    if amt > 200_000:
        f.append("Amount bahut bada hai (> 2,00,000).")
    return f or ["Koi bada rule-based red flag nahi mila."]


FASTAPI_CODE = '''"""FastAPI service for the fraud model.  Run:  uvicorn api.main:app --reload --port 8000"""
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
'''

DOCKERFILE_APP = '''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
'''

DOCKERFILE_API = '''FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pandas numpy scikit-learn joblib xgboost
COPY core ./core
COPY api ./api
COPY models ./models
ENV MODEL_PATH=models/fraud_model.joblib
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''

CURL_EXAMPLE = '''curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"type":"TRANSFER","amount":181000,"oldbalanceOrg":181000,"newbalanceOrig":0,
       "oldbalanceDest":0,"newbalanceDest":0}'
'''
