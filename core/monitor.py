"""Monitoring: drift (PSI), retraining triggers."""
import numpy as np
import pandas as pd


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index: <0.1 stable, 0.1–0.25 moderate shift, >0.25 big shift."""
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    if len(e) == 0 or len(a) == 0:
        return 0.0
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    ep = np.clip(np.histogram(e, qs)[0] / len(e), 1e-4, None)
    ap = np.clip(np.histogram(a, qs)[0] / len(a), 1e-4, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def drift_report(ref: pd.DataFrame, new: pd.DataFrame, features) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in ref.columns and f in new.columns:
            v = psi(ref[f], new[f])
            rows.append({"feature": f, "PSI": round(v, 4),
                         "status": "Stable ✅" if v < 0.1 else ("Moderate ⚠️" if v < 0.25 else "High 🚨")})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def simulate_drift(df: pd.DataFrame, features, strength: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Demo only: shift some features to imitate changing customer behaviour."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    for f in features:
        if d[f].nunique() > 2 and rng.random() < 0.5:
            d[f] = d[f] * (1 + strength * rng.random()) + strength * d[f].std()
    return d


def retrain_decision(history: pd.DataFrame, min_recall=0.80, min_pr_auc=0.60, max_psi=0.25, worst_psi=0.0):
    """Return (should_retrain, reasons)."""
    reasons = []
    if len(history):
        last = history.iloc[-1]
        if last["recall"] < min_recall:
            reasons.append(f"Recall {last['recall']:.2f} < {min_recall:.2f}")
        if last["pr_auc"] < min_pr_auc:
            reasons.append(f"PR-AUC {last['pr_auc']:.2f} < {min_pr_auc:.2f}")
    if worst_psi > max_psi:
        reasons.append(f"Data drift (max PSI {worst_psi:.2f} > {max_psi:.2f})")
    return bool(reasons), reasons
