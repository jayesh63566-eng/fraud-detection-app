"""Splitting, imbalance handling (train only!), model zoo, training, CV and tuning."""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.evaluation import compute_metrics

RESAMPLE_METHODS = {
    "None": "Kuch nahi karna (original imbalanced data).",
    "Class weights": "Model ko fraud rows par zyada weight dene ko kehna. Data nahi badalta, sabse safe.",
    "SMOTE": "Fraud ke naye synthetic examples banata hai (neighbours ke beech mein).",
    "Random oversample": "Fraud rows ko copy karke badhata hai.",
    "Random undersample": "Normal rows kam kar deta hai (data ka bahut hissa phenk deta hai).",
}


# ------------------------------------------------------------------ split
def make_split(df: pd.DataFrame, features: list[str], test_size=0.2, seed=42, mode="Stratified random"):
    X = df[features]
    y = df["isFraud"].astype(int)
    if mode == "Time-based (step)" and "step" in df.columns:
        cutoff = df["step"].quantile(1 - test_size)
        test_mask = df["step"] > cutoff
        Xtr, Xte, ytr, yte = X[~test_mask], X[test_mask], y[~test_mask], y[test_mask]
    else:
        strat = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    return dict(X_train=Xtr, X_test=Xte, y_train=ytr, y_test=yte, features=list(features),
                mode=mode, test_size=test_size, seed=seed)


# ------------------------------------------------------------------ resampling
def _smote(X: np.ndarray, y: np.ndarray, ratio: float, seed: int, k: int = 5):
    rng = np.random.default_rng(seed)
    Xm = X[y == 1]
    n_min, n_maj = len(Xm), int((y == 0).sum())
    n_new = int(ratio * n_maj) - n_min
    if n_new <= 0 or n_min < 2:
        return X, y
    k = min(k, n_min - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, n_min, n_new)
    nb = idx[base, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1))
    Xs = Xm[base] + lam * (Xm[nb] - Xm[base])
    return np.vstack([X, Xs]), np.concatenate([y, np.ones(n_new, dtype=int)])


def resample(X: pd.DataFrame, y: pd.Series, method: str, ratio: float = 0.5, seed: int = 42):
    """Applied to the TRAIN part only. `ratio` = minority/majority after resampling."""
    cols = X.columns
    Xa, ya = X.to_numpy(float), np.asarray(y).astype(int)
    if method in ("None", "Class weights") or ya.sum() < 2:
        return X, pd.Series(ya, index=X.index)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(ya == 1)[0], np.where(ya == 0)[0]
    if method == "SMOTE":
        try:
            from imblearn.over_sampling import SMOTE
            Xr, yr = SMOTE(sampling_strategy=ratio, random_state=seed,
                           k_neighbors=min(5, len(pos) - 1)).fit_resample(Xa, ya)
        except ImportError:
            Xr, yr = _smote(Xa, ya, ratio, seed)
    elif method == "Random oversample":
        n_new = max(int(ratio * len(neg)) - len(pos), 0)
        extra = rng.choice(pos, n_new, replace=True)
        sel = np.concatenate([np.arange(len(ya)), extra])
        Xr, yr = Xa[sel], ya[sel]
    elif method == "Random undersample":
        keep_neg = rng.choice(neg, min(len(neg), int(len(pos) / ratio)), replace=False)
        sel = np.concatenate([pos, keep_neg])
        Xr, yr = Xa[sel], ya[sel]
    else:
        raise ValueError(method)
    perm = rng.permutation(len(yr))
    return pd.DataFrame(Xr[perm], columns=cols), pd.Series(yr[perm])


# ------------------------------------------------------------------ model zoo
def _zoo():
    zoo = {
        "Logistic Regression": dict(
            make=lambda cw: LogisticRegression(max_iter=1000, class_weight=cw), scale=True,
            desc="Simple linear model. Tez, samajhne mein aasaan (baseline)."),
        "Decision Tree": dict(
            make=lambda cw: DecisionTreeClassifier(max_depth=12, class_weight=cw, random_state=42), scale=False,
            desc="Yes/No sawaalon ka tree. Interpret karna aasaan."),
        "Random Forest": dict(
            make=lambda cw: RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight=cw, random_state=42),
            scale=False, desc="Bahut saare trees ka vote. Fraud detection ka strong default."),
        "Gradient Boosting": dict(
            make=lambda cw: GradientBoostingClassifier(random_state=42), scale=False,
            desc="Trees ek-ek karke pichli galtiyan sudharte hain. Accurate par thoda slow."),
        "KNN": dict(
            make=lambda cw: KNeighborsClassifier(n_neighbors=5, n_jobs=-1), scale=True,
            desc="Sabse milte-julte 5 transactions dekh kar faisla. Bade data par slow."),
        "SVC": dict(
            make=lambda cw: SVC(probability=True, class_weight=cw, random_state=42), scale=True,
            desc="Support Vector Machine. Slow, isliye 20k rows tak subsample hota hai."),
    }
    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = dict(
            make=lambda cw: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          eval_metric="logloss", n_jobs=-1, random_state=42),
            scale=False, desc="Boosting ka fast, powerful version. Kaggle favourite.")
    except Exception:
        pass
    return zoo


MODEL_ZOO = _zoo()
DEFAULT_MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]


def build_pipeline(name: str, class_weight=None) -> Pipeline:
    z = MODEL_ZOO[name]
    steps = []
    if z["scale"]:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", z["make"](class_weight)))
    return Pipeline(steps)


def _cap_rows(X, y, cap, seed):
    """Keep all frauds, subsample normals so total <= cap (for slow models)."""
    if len(X) <= cap:
        return X, y
    y = pd.Series(np.asarray(y), index=X.index)
    pos = y[y == 1].index
    neg = y[y == 0].sample(max(cap - len(pos), 1), random_state=seed).index
    keep = pos.union(neg)
    return X.loc[keep], y.loc[keep]


def train_model(name: str, split: dict, method="Class weights", ratio=0.5, seed=42):
    t0 = time.time()
    Xtr, ytr = split["X_train"], split["y_train"]
    cw = "balanced" if method == "Class weights" else None
    Xr, yr = resample(Xtr, ytr, method, ratio, seed)
    if name in ("SVC", "KNN"):
        cap = 20000 if name == "SVC" else 100000
        Xr, yr = _cap_rows(Xr.reset_index(drop=True), yr.reset_index(drop=True), cap, seed)
    pipe = build_pipeline(name, cw)
    if name == "XGBoost" and method == "Class weights" and (yr == 1).sum() > 0:
        pipe.set_params(model__scale_pos_weight=float((yr == 0).sum() / max((yr == 1).sum(), 1)))
    pipe.fit(Xr, yr)
    prob = pipe.predict_proba(split["X_test"])[:, 1]
    m = compute_metrics(split["y_test"], prob, 0.5)
    return dict(name=name, pipeline=pipe, y_prob=prob, metrics=m, train_seconds=round(time.time() - t0, 2),
                method=method, ratio=ratio, n_train_used=int(len(Xr)))


# ------------------------------------------------------------------ cross-validation
def cross_validate_model(name, X, y, folds=5, method="Class weights", ratio=0.5, seed=42, progress=None):
    """Stratified K-fold. Resampling is done INSIDE each training fold (no leakage)."""
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y).astype(int))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        Xr, yr = resample(X.iloc[tr], y.iloc[tr], method, ratio, seed)
        if name in ("SVC", "KNN"):
            Xr, yr = _cap_rows(Xr.reset_index(drop=True), yr.reset_index(drop=True),
                               20000 if name == "SVC" else 100000, seed)
        pipe = build_pipeline(name, "balanced" if method == "Class weights" else None).fit(Xr, yr)
        prob = pipe.predict_proba(X.iloc[va])[:, 1]
        m = compute_metrics(y.iloc[va], prob, 0.5)
        rows.append(dict(fold=i, precision=m["precision"], recall=m["recall"], f1=m["f1"],
                         roc_auc=m["roc_auc"], pr_auc=m["pr_auc"]))
        if progress:
            progress(i / folds)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tuning
PARAM_GRIDS = {
    "Random Forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 8, 12, 20],
                      "model__min_samples_leaf": [1, 2, 5], "model__max_features": ["sqrt", "log2"]},
    "Decision Tree": {"model__max_depth": [4, 6, 8, 12, 16, None], "model__min_samples_leaf": [1, 2, 5, 10],
                      "model__criterion": ["gini", "entropy"]},
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l2"]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1, 0.2],
                          "model__max_depth": [2, 3, 5]},
    "KNN": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"]},
    "SVC": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
    "XGBoost": {"model__n_estimators": [100, 200, 300], "model__max_depth": [3, 6, 9],
                "model__learning_rate": [0.03, 0.1, 0.3], "model__subsample": [0.7, 1.0]},
}


def tune_model(name, split, n_iter=8, cv=3, scoring="average_precision", cap=30000, seed=42):
    """RandomizedSearchCV on (a capped sample of) the training data, class-weight based."""
    X, y = _cap_rows(split["X_train"], split["y_train"], cap, seed)
    pipe = build_pipeline(name, "balanced")
    grid = PARAM_GRIDS[name]
    n_iter = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
    rs = RandomizedSearchCV(pipe, grid, n_iter=n_iter, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                            scoring=scoring, n_jobs=-1, random_state=seed, refit=True)
    rs.fit(X, y)
    best = rs.best_estimator_
    prob = best.predict_proba(split["X_test"])[:, 1]
    res = pd.DataFrame(rs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    return dict(name=f"{name} (tuned)", pipeline=best, y_prob=prob,
                metrics=compute_metrics(split["y_test"], prob, 0.5), best_params=rs.best_params_,
                cv_score=float(rs.best_score_), cv_table=res.sort_values("rank_test_score"),
                method="Class weights", ratio=0.5, train_seconds=0.0, n_train_used=len(X))
