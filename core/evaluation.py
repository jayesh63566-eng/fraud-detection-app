"""Metrics, threshold analysis and business-cost analysis (no plotting here)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

METRIC_HELP = {
    "precision": "Jitne transactions ko model ne 'fraud' kaha, unme se kitne sach mein fraud the. (kam false alarm)",
    "recall": "Jitne asli fraud the, unme se kitne model ne pakde. (kam fraud miss)",
    "f1": "Precision aur recall ka balance.",
    "pr_auc": "Precision-Recall curve ke neeche ka area. Imbalanced data ke liye sabse bharosemand number.",
    "roc_auc": "Model kitna achha fraud aur normal ko alag karta hai (0.5 = guess, 1.0 = perfect).",
    "accuracy": "Kul sahi predictions. ⚠️ Fraud 1% ho to 'sab normal' bolne se bhi 99% milta hai, isliye akela accuracy dhokha hai.",
    "mcc": "Matthews correlation: imbalanced data ke liye ek balanced score (−1 se +1).",
    "balanced_accuracy": "Fraud aur normal dono classes ka average recall.",
}


def compute_metrics(y_true, y_prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two = len(np.unique(y_true)) == 2
    return dict(
        threshold=float(thr),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 or two else 0.0,
        roc_auc=roc_auc_score(y_true, y_prob) if two else float("nan"),
        pr_auc=average_precision_score(y_true, y_prob) if two else float("nan"),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def threshold_table(y_true, y_prob, fn_cost=1000.0, fp_cost=10.0, points=99) -> pd.DataFrame:
    """Precision / recall / F1 and total business cost for each decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rows = []
    for t in np.linspace(0.01, 0.99, points):
        p = y_prob >= t
        tp = int((p & (y_true == 1)).sum())
        fp = int((p & (y_true == 0)).sum())
        fn = int((~p & (y_true == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(threshold=round(float(t), 2), precision=prec, recall=rec, f1=f1,
                         tp=tp, fp=fp, fn=fn, cost=fn * fn_cost + fp * fp_cost))
    return pd.DataFrame(rows)


def best_threshold(tbl: pd.DataFrame, by="f1") -> float:
    return float(tbl.loc[tbl["cost"].idxmin() if by == "cost" else tbl[by].idxmax(), "threshold"])


def leaderboard(models: dict) -> pd.DataFrame:
    rows = []
    for name, r in models.items():
        m = r["metrics"]
        rows.append({"model": name, "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "recall": m["recall"],
                     "precision": m["precision"], "f1": m["f1"], "mcc": m["mcc"], "accuracy": m["accuracy"],
                     "train_s": r.get("train_seconds", 0.0)})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
