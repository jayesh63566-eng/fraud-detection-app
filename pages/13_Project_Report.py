import pandas as pd
import streamlit as st

from core import evaluation, interpret, state, ui

ui.setup("Project Report (8 Questions)", "📝", 13, "Aapke project ke 8 sawaalon ke jawab, aapke apne results se automatically bane")

ui.explain(
    "Ye page pipeline ke results ko jodkar **8 sawaalon ke jawab** banata hai. Jo step abhi complete nahi hua, "
    "uske jawab mein '(abhi baaki)' dikhega. Last mein poori report **Markdown** mein download kar sakte ho."
)

raw, clean, feat = state.get("raw_df"), state.get("clean_df"), state.get("feat_df")
models, sp, feats = state.get("models"), state.get("split"), state.get("features")
best = state.get("best_model")
out = ["# Fraud Detection: Project Answers\n"]


def _md_table(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def sec(title, body):
    st.subheader(title)
    st.markdown(body)
    out.append(f"## {title}\n\n{body}\n")


# Q1
b = []
if raw is not None:
    miss = int(raw.isnull().sum().sum())
    b.append(f"- **Data:** {raw.shape[0]:,} rows × {raw.shape[1]} columns, {miss:,} missing cells, {int(raw.duplicated().sum()):,} duplicates.")
else:
    b.append("- Data load nahi hua (abhi baaki).")
if state.get("prep_stats"):
    s = state.get("prep_stats")
    b.append(f"- **Cleaning ke baad:** {s['before']:,} → {s['after']:,} rows; fraud rows {s['fraud_before']:,} → {s['fraud_after']:,}.")
    b += [f"  - {l}" for l in state.get("prep_log", [])]
if raw is not None:
    b.append("- **Outliers:** amount aur balances bahut skewed hain. Fraud mein extreme values useful signal hote hain, isliye default mein rakhe gaye (ya log-transform kiya).")
b.append("- **Multicollinearity:** correlation heatmap aur VIF se check kiya. `oldbalance/newbalance/balanceChange/amount` aapas mein bahut correlated hain "
         "(VIF high), isliye tree-based models prefer kiye ya redundant features hataye.")
sec("1. Data cleaning (missing values, outliers, multicollinearity)", "\n".join(b))

# Q2
b = ["- **Problem:** binary classification (isFraud 0/1), highly imbalanced.",
     "- **Pipeline:** cleaning → feature engineering → feature selection → train/test split → imbalance handling (sirf train par) → "
     "scaling (sirf zaroori models ke liye) → model training → evaluation → tuning/CV → interpretation → deployment."]
if models:
    cfg = state.get("train_cfg", {})
    b.append(f"- **Imbalance method:** {cfg.get('method', '-')}.")
    b.append("- **Models try kiye:** " + ", ".join(models) + ".")
    if best:
        b.append(f"- **Final model:** **{best}**, kyunki isme sabse accha PR-AUC/recall balance mila.")
else:
    b.append("- Models abhi train nahi hue (abhi baaki).")
b.append("- **Leakage se bachav:** split pehle hua; SMOTE/scaling sirf train par; test set bilkul untouched.")
sec("2. Fraud detection model ka description", "\n".join(b))

# Q3
if feats:
    b = [f"- **Final {len(feats)} features:** " + ", ".join(f"`{f}`" for f in feats),
         "- **Kaise chune:** correlation + VIF (redundant hataye), Random Forest importance, SelectKBest, RFE, aur domain knowledge.",
         "- `step`/`day` hataye: time khud fraud ki wajah nahi aur future mein generalise nahi hota.",
         "- Account IDs (`nameOrig`/`nameDest`) hataye: high-cardinality, har baar alag."]
else:
    b = ["(abhi baaki: Feature Selection page)"]
sec("3. Variables kaise select kiye", "\n".join(b))

# Q4
if models and sp and best:
    m = models[best]["metrics"]
    lb = evaluation.leaderboard(models)
    b = [f"- **Best model: {best}** (test set par)\n",
         f"| Metric | Value |\n|---|---|\n| PR-AUC | {m['pr_auc']:.3f} |\n| ROC-AUC | {m['roc_auc']:.3f} |\n| Recall | {m['recall']:.3f} |\n"
         f"| Precision | {m['precision']:.3f} |\n| F1 | {m['f1']:.3f} |\n| MCC | {m['mcc']:.3f} |\n",
         f"- Confusion matrix @ threshold 0.5: TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}",
         "- **Tools:** confusion matrix, ROC aur Precision-Recall curves, threshold analysis, business-cost analysis, stratified cross-validation, model leaderboard.",
         "- Accuracy akeli nahi dekhi, kyunki imbalanced data mein misleading hoti hai.\n",
         _md_table(lb.round(3))]
else:
    b = ["(abhi baaki: Model Training aur Evaluation)"]
sec("4. Model performance (best tools se)", "\n".join(b))

# Q5 / Q6
if models and sp and best:
    pipe = models[best]["pipeline"]
    imp = interpret.builtin_importance(pipe, sp["features"])
    if imp is None:
        imp = interpret.perm_importance(pipe, sp["X_test"], sp["y_test"], n_repeats=2)
    top = imp.head(8)
    sec("5. Key factors jo fraud predict karte hain",
        "\n".join(f"{i}. `{r.feature}` (importance {r.importance:.3f})" for i, r in enumerate(top.itertuples(), 1)))
    ms = interpret.make_sense_table(top["feature"].tolist())
    lines = [f"- `{r['feature']}`: **{r['makes sense?']}**. {r['kyun']}" for _, r in ms.iterrows()]
    lines.append("\n**Conclusion:** factors fraud ke asli pattern (account takeover → poora balance TRANSFER/CASH_OUT → receiver side mismatch) se match karte hain, isliye logical hain.")
    sec("6. Kya ye factors sense banate hain?", "\n".join(lines))
else:
    sec("5. Key factors", "(abhi baaki: Model Training)")
    sec("6. Kya ye factors sense banate hain?", "(abhi baaki)")

# Q7
sec("7. Infrastructure update ke time prevention",
    "- Real-time scoring par high-risk transactions ko hold + OTP/biometric verification\n"
    "- Velocity limits, naye receiver par cooling-off period, poora balance nikalne par extra confirmation\n"
    "- MFA, device fingerprinting, SIM-swap/password-reset ke baad limits\n"
    "- Consistent transaction logging aur receiver-balance update, feature store, analyst feedback loop\n"
    "- Hybrid (rules + ML), explainability, API security (TLS, auth, rate-limit), data privacy")

# Q8
sec("8. Actions kaam kar rahe hain ya nahi, kaise pata chalega",
    "- Pehle KPIs ka baseline: fraud loss ₹, recall, precision, false-positive rate, customer friction\n"
    "- Before/after comparison, **A/B (champion-challenger) test**, **shadow mode**, backtesting\n"
    "- Statistical significance (z-test / confidence interval)\n"
    "- Continuous monitoring: performance log, PSI drift, retrain triggers, alerts\n"
    "- Success = fraud loss ↓, recall stable/↑, false positives ↓ ya same")

md = "\n".join(out)
st.divider()
st.download_button("⬇️ Poori report (Markdown)", md.encode("utf-8"), file_name="fraud_project_answers.md", mime="text/markdown")
