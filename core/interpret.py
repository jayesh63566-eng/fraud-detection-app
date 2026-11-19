"""Model interpretation: built-in + permutation importance, local (occlusion) explanations,
optional SHAP / LIME, and domain-knowledge sanity checks."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

DOMAIN_NOTES = {
    "errorBalanceOrig": ("Haan ✅", "Sahi transaction mein old − amount = new hota hai. Mismatch ka matlab balance ke saath chhed-chhaad / abnormal flow."),
    "errorBalanceDest": ("Haan ✅", "Receiver ke balance mein amount add nahi hua to paisa kahin aur gaya (mule / hidden account)."),
    "accountEmptied": ("Haan ✅", "Fraudster aam taur par poora account khaali karta hai (account takeover ka classic pattern)."),
    "fullBalanceTransfer": ("Haan ✅", "Poora balance ek hi baar mein bhej dena normal customer ke liye kam hota hai."),
    "newbalanceOrig": ("Haan ✅", "Transaction ke baad sender balance 0 = account drain hua."),
    "oldbalanceOrg": ("Haan ✅", "Fraudster tab attack karta hai jab account mein achha balance ho."),
    "amount": ("Haan ✅", "Bade amount = zyada fayda fraudster ko, isliye risk zyada."),
    "amount_log": ("Haan ✅", "Amount ka hi transformed roop; same logic."),
    "isHighAmount": ("Haan ✅", "Bahut bade transfer par risk zyada hota hai."),
    "type_TRANSFER": ("Haan ✅", "Fraud sirf TRANSFER aur CASH_OUT mein hota hai (paisa nikalne ke raaste)."),
    "type_CASH_OUT": ("Haan ✅", "Fraudster paisa cash mein nikalta hai. Ye TRANSFER ke baad ka step hota hai."),
    "type_PAYMENT": ("Haan ✅", "PAYMENT mein fraud lagbhag nahi hota, isliye ye feature 'normal' ka signal hai."),
    "type_CASH_IN": ("Haan ✅", "CASH_IN mein fraud nahi hota; negative signal."),
    "type_DEBIT": ("Haan ✅", "DEBIT mein fraud bahut kam hota hai."),
    "destBalanceUnchanged": ("Haan ✅", "Receiver ka balance nahi badla = paisa system ke bahar ya merchant account mein gaya."),
    "oldbalanceDest": ("Aanshik ⚠️", "Receiver ka purana balance 0 hona naye/mule account ka signal ho sakta hai."),
    "newbalanceDest": ("Aanshik ⚠️", "Akele mein kamzor signal, errorBalanceDest ke saath mil kar kaam karta hai."),
    "balanceChangeOrig": ("Haan ✅", "Sender balance mein bada girawat = drain."),
    "balanceChangeDest": ("Haan ✅", "Receiver balance mein badlav na hona/bahut zyada hona dono suspicious."),
    "isMerchantDest": ("Haan ✅", "Merchant ko bheja paisa (PAYMENT) aam taur par safe; customer-to-customer risky."),
    "hour_of_day": ("Aanshik ⚠️", "Raat ke ghanton mein fraud zyada ho sakta hai, par simulated data mein ye weak hota hai."),
    "isNight": ("Aanshik ⚠️", "Raat mein monitoring kam hoti hai, isliye fraudsters ka favourite time."),
    "day": ("Nahi ❌", "Date khud fraud ki wajah nahi. Future data mein generalise nahi hota; isliye default mein hata diya."),
    "step": ("Nahi ❌", "Time index khud fraud ki wajah nahi. Overfitting ka khatra."),
    "senderTxnCount": ("Haan ✅", "Naye ya bahut active sender ka behaviour alag hota hai (velocity)."),
    "receiverTxnCount": ("Haan ✅", "Ek receiver ko bahut saare transfer = mule account ka signal."),
}


def builtin_importance(pipe, features):
    est = pipe.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
        kind = "Tree importance"
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_).ravel()
        kind = "|Coefficient| (scaled)"
    else:
        return None
    df = pd.DataFrame({"feature": features, "importance": vals}).sort_values("importance", ascending=False)
    df["kind"] = kind
    return df.reset_index(drop=True)


def _balanced_sample(X, y, n=4000, seed=42):
    y = pd.Series(np.asarray(y), index=X.index)
    pos = y[y == 1].index
    pos = pos[: n // 2] if len(pos) > n // 2 else pos
    neg = y[y == 0].sample(min(n - len(pos), int((y == 0).sum())), random_state=seed).index
    idx = pos.union(neg)
    return X.loc[idx], y.loc[idx]


def perm_importance(pipe, X, y, n_repeats=5, sample=4000, scoring="average_precision", seed=42):
    Xs, ys = _balanced_sample(X, y, sample, seed)
    r = permutation_importance(pipe, Xs, ys, n_repeats=n_repeats, scoring=scoring, random_state=seed, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def occlusion_explain(pipe, row: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """For each feature: replace it with a typical (median) value and see how the fraud probability moves.
    Positive effect = this feature pushed the score UP."""
    p0 = float(pipe.predict_proba(row)[0, 1])
    rows = []
    for c in row.columns:
        r2 = row.copy()
        r2[c] = baseline[c]
        rows.append(dict(feature=c, value=float(row[c].iloc[0]), typical=float(baseline[c]),
                         effect=p0 - float(pipe.predict_proba(r2)[0, 1])))
    out = pd.DataFrame(rows)
    out["abs"] = out["effect"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True), p0


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def lime_available() -> bool:
    try:
        import lime  # noqa: F401
        return True
    except Exception:
        return False


def shap_values_tree(pipe, X: pd.DataFrame):
    """Return (shap_values[n, f], X_used). Works for tree models."""
    import shap
    est = pipe.named_steps["model"]
    Xs = pipe[:-1].transform(X) if len(pipe.steps) > 1 else X
    Xs = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    ex = shap.TreeExplainer(est)
    sv = ex.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, Xs


def lime_explain(pipe, X_train: pd.DataFrame, row: pd.DataFrame, num_features=8):
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(X_train.to_numpy(float), feature_names=list(X_train.columns),
                              class_names=["normal", "fraud"], mode="classification", discretize_continuous=True)
    fn = lambda a: pipe.predict_proba(pd.DataFrame(a, columns=X_train.columns))
    e = ex.explain_instance(row.to_numpy(float).ravel(), fn, num_features=num_features)
    return pd.DataFrame(e.as_list(), columns=["rule", "weight"])


def make_sense_table(top_features) -> pd.DataFrame:
    rows = []
    for f in top_features:
        verdict, why = DOMAIN_NOTES.get(f, ("Check karo ❓", "Is feature ke liye domain note nahi hai. Business team se confirm karo."))
        rows.append({"feature": f, "makes sense?": verdict, "kyun": why})
    return pd.DataFrame(rows)
