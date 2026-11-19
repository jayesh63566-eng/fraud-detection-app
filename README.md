# Fraud Detection System (Streamlit)

MSc Data Science project: end-to-end fraud detection workflow in a web app.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the sidebar → **"⚡ Demo data se shuru karo"**, or go to *Data Collection* for your own CSV / Google Drive / Kaggle API.

## Pages
1 Data Collection · 2 EDA · 3 Visualization · 4 Preprocessing · 5 Feature Engineering · 6 Feature Selection ·
7 Model Training · 8 Evaluation · 9 Tuning & CV · 10 Interpretation · 11 Deployment · 12 Monitoring & Prevention · 13 Project Report

## Serve the model as an API
Save a model bundle on the *Deployment* page (writes `models/fraud_model.joblib`), then:
```bash
uvicorn api.main:app --port 8000      # docs at http://localhost:8000/docs
```

## Docker
```bash
docker build -t fraud-app . && docker run -p 8501:8501 fraud-app
docker build -f Dockerfile.api -t fraud-api . && docker run -p 8000:8000 fraud-api
```

## Design notes
- Split happens **before** SMOTE / scaling; resampling touches the train set only (no leakage).
- Same `core.features.engineer()` is used for training and live prediction (no train/serve skew).
- Optional libs (xgboost, shap, lime, imblearn, gdown) are detected at runtime; the app degrades gracefully.
- Tests: `python tests/test_core.py`
