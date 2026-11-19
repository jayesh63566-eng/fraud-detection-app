"""Preprocessing helpers: missing values, duplicates, outliers, multicollinearity."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isnull().sum()
    out = pd.DataFrame({"column": m.index, "missing": m.values,
                        "missing_%": (m.values / max(len(df), 1) * 100).round(3)})
    return out.sort_values("missing", ascending=False).reset_index(drop=True)


def impute(df: pd.DataFrame, num_strategy="median", cat_strategy="most_frequent", drop_rows=False):
    """Returns (df, log). num_strategy: mean/median/zero ; cat_strategy: most_frequent/unknown."""
    d = df.copy()
    log = []
    before = int(d.isnull().sum().sum())
    if before == 0:
        return d, ["Koi missing value nahi mili."]
    if drop_rows:
        n = len(d)
        d = d.dropna()
        return d, [f"{n - len(d)} rows drop kiye (missing values wale)."]
    for c in d.columns:
        if d[c].isnull().any():
            if pd.api.types.is_numeric_dtype(d[c]):
                val = {"mean": d[c].mean(), "median": d[c].median(), "zero": 0}[num_strategy]
                d[c] = d[c].fillna(val)
                log.append(f"'{c}': {num_strategy} se fill kiya.")
            else:
                val = d[c].mode().iloc[0] if cat_strategy == "most_frequent" and not d[c].mode().empty else "UNKNOWN"
                d[c] = d[c].fillna(val)
                log.append(f"'{c}': '{val}' se fill kiya.")
    return d, log


def outlier_bounds(s: pd.Series, method="IQR", k=1.5, z=3.0, pct=(0.01, 0.99)):
    s = s.dropna()
    if method == "IQR":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return q1 - k * iqr, q3 + k * iqr
    if method == "Z-score":
        return s.mean() - z * s.std(), s.mean() + z * s.std()
    return s.quantile(pct[0]), s.quantile(pct[1])


def outlier_report(df: pd.DataFrame, cols, method="IQR", **kw) -> pd.DataFrame:
    rows = []
    for c in cols:
        lo, hi = outlier_bounds(df[c], method, **kw)
        mask = (df[c] < lo) | (df[c] > hi)
        fraud_share = np.nan
        if "isFraud" in df.columns and mask.sum() > 0:
            fraud_share = round(float(df.loc[mask, "isFraud"].mean() * 100), 2)
        rows.append({"column": c, "lower": round(float(lo), 2), "upper": round(float(hi), 2),
                     "outliers": int(mask.sum()), "outlier_%": round(mask.mean() * 100, 2),
                     "fraud_% inside outliers": fraud_share})
    return pd.DataFrame(rows)


def handle_outliers(df: pd.DataFrame, cols, method="IQR", action="keep", **kw):
    """action: keep | cap | remove. Returns (df, log)."""
    d = df.copy()
    log = []
    if action == "keep" or not cols:
        return d, ["Outliers rakhe gaye (fraud mein extreme values useful signal hote hain)."]
    for c in cols:
        lo, hi = outlier_bounds(d[c], method, **kw)
        if action == "cap":
            n = int(((d[c] < lo) | (d[c] > hi)).sum())
            d[c] = d[c].clip(lo, hi)
            log.append(f"'{c}': {n} values [{lo:,.0f}, {hi:,.0f}] par cap ki.")
        elif action == "remove":
            n0 = len(d)
            d = d[(d[c] >= lo) & (d[c] <= hi)]
            log.append(f"'{c}': {n0 - len(d)} rows hataye.")
    return d, log


def vif_table(df: pd.DataFrame, cols, max_rows=20000, seed=42) -> pd.DataFrame:
    """Variance Inflation Factor via R² of each column regressed on the others (no statsmodels needed)."""
    X = df[cols].astype(float)
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    X = X.loc[:, X.std() > 0]
    out = []
    for c in X.columns:
        others = [o for o in X.columns if o != c]
        if not others:
            out.append((c, 1.0))
            continue
        r2 = LinearRegression().fit(X[others], X[c]).score(X[others], X[c])
        out.append((c, np.inf if r2 >= 0.99999 else 1.0 / (1.0 - r2)))
    t = pd.DataFrame(out, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
    t["verdict"] = pd.cut(t["VIF"], [-np.inf, 5, 10, np.inf], labels=["OK", "Moderate", "High (drop?)"])
    return t.reset_index(drop=True)


def high_corr_pairs(df: pd.DataFrame, cols, threshold=0.9) -> pd.DataFrame:
    c = df[cols].corr().abs()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = c.loc[a, b]
            if v >= threshold:
                rows.append({"feature_1": a, "feature_2": b, "|corr|": round(float(v), 4)})
    return pd.DataFrame(rows).sort_values("|corr|", ascending=False) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "|corr|"])
