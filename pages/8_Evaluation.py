import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report

from core import evaluation, plots, state, ui

ui.setup("Model Evaluation", "📈", 8, "Model kitna accha hai? Sahi tools se napo")
models = state.require("models", "Model Training", "pages/7_Model_Training.py")
sp = state.require("split", "Model Training", "pages/7_Model_Training.py")
y = sp["y_test"].to_numpy()

ui.explain(
    "**Accuracy yahan dhokha deti hai.** Fraud 1% ho to 'sab normal' bolne se 99% accuracy milti hai. Isliye 4 cheezein dekhte hain:\n\n"
    "- **Recall** = asli fraud mein se kitne pakde\n"
    "- **Precision** = jo fraud bola, usme se kitne sach mein fraud\n"
    "- **PR-AUC** = in dono ka overall score (best single number)\n"
    "- **Confusion matrix** = sahi/galat ka poora hisaab\n\n"
    "Aur **threshold** (kitni probability par 'fraud' bolna hai) tum khud badal ke dekh sakte ho."
)

lb = evaluation.leaderboard(models)
names = list(models)
best_default = state.get("best_model") if state.get("best_model") in names else names[0]
name = st.selectbox("Model chuno", names, index=names.index(best_default))
r = models[name]
prob = r["y_prob"]

tabs = st.tabs(["🎯 Metrics & Threshold", "📉 Curves", "🏆 Compare models", "💰 Business cost", "🔎 Errors"])

with tabs[0]:
    thr = st.slider("Decision threshold", 0.01, 0.99, float(state.get("threshold", 0.5)), 0.01,
                    help="Kam threshold = zyada fraud pakdoge (recall ↑) par zyada false alarm (precision ↓).")
    state.put("threshold", thr)
    m = evaluation.compute_metrics(y, prob, thr)
    ui.metric_row(m, ("pr_auc", "recall", "precision", "f1", "roc_auc"))
    a, b = st.columns([1, 1])
    with a:
        ui.plot(plots.confusion_fig(m))
    with b:
        st.markdown(f"**Is threshold ({thr:.2f}) par:**")
        st.markdown(
            f"- ✅ **{m['tp']:,}** fraud pakde, ❌ **{m['fn']:,}** miss hue\n"
            f"- 🚨 **{m['fp']:,}** false alarm (normal ko fraud bola)\n"
            f"- Accuracy: {m['accuracy']:.3f} (⚠️ akeli bharosemand nahi)"
        )
        with st.expander("Metrics ka matlab"):
            for k, v in evaluation.METRIC_HELP.items():
                st.markdown(f"**{k}**: {v}")
    tbl = evaluation.threshold_table(y, prob)
    ui.plot(plots.threshold_fig(tbl, thr))
    c1, c2 = st.columns(2)
    if c1.button("🎯 Best F1 threshold lagao"):
        state.put("threshold", evaluation.best_threshold(tbl, "f1"))
        st.rerun()
    c2.caption("Recall ≥ 90% chahiye? Slider ko tab tak neeche karo jab tak recall 0.9 na ho jaye.")
    st.markdown("**Classification report**")
    yp = (prob >= thr).astype(int)
    ui.table(pd.DataFrame(classification_report(y, yp, target_names=["Normal", "Fraud"], output_dict=True, zero_division=0)).T.round(3).reset_index().rename(columns={"index": "class"}))

with tabs[1]:
    sel = st.multiselect("Kaun se models overlay karne hain", names, default=names)
    sub = {k: models[k] for k in sel}
    if sub:
        a, b = st.columns(2)
        with a:
            ui.plot(plots.roc_fig(sub, y))
        with b:
            ui.plot(plots.pr_fig(sub, y))
        st.caption("PR curve jitni upar-right mein, utna accha model. Imbalanced data mein ROC ziada optimistic dikhta hai, isliye PR curve par zyada bharosa karo.")

with tabs[2]:
    ui.table(lb.round(4))
    ui.plot(plots.leaderboard_fig(lb))
    best = st.selectbox("⭐ Deployment ke liye best model chuno", names, index=names.index(best_default), key="pick_best")
    if st.button("⭐ Ye best model hai"):
        state.put("best_model", best)
        st.success(f"'{best}' best model set ho gaya.")

with tabs[3]:
    st.markdown("**Fraud miss karna aur false alarm dono ka kharcha alag hota hai.** Apne numbers daalo:")
    c1, c2 = st.columns(2)
    fn_c = c1.number_input("Ek missed fraud ka nuksaan (₹)", 0, 10_000_000, 10000, 500)
    fp_c = c2.number_input("Ek false alarm ka kharcha (₹, review + customer pareshani)", 0, 1_000_000, 50, 10)
    tbl = evaluation.threshold_table(y, prob, fn_c, fp_c)
    cur = float(state.get("threshold", 0.5))
    ui.plot(plots.cost_fig(tbl, cur))
    bt = evaluation.best_threshold(tbl, "cost")
    row = tbl.loc[tbl["threshold"] == bt].iloc[0]
    st.success(f"Sabse kam kharcha **threshold {bt:.2f}** par: ₹{row['cost']:,.0f} (fraud miss: {int(row['fn'])}, false alarm: {int(row['fp'])})")
    base = evaluation.threshold_table(y, prob, fn_c, fp_c).iloc[-1]
    no_model = int(y.sum()) * fn_c
    st.metric("Bina model ke nuksaan (saare fraud miss)", f"₹{no_model:,.0f}", f"-₹{no_model - row['cost']:,.0f} bachat model se")
    if st.button("💰 Ye cost-optimal threshold use karo"):
        state.put("threshold", bt)
        st.rerun()

with tabs[4]:
    thr = float(state.get("threshold", 0.5))
    X = sp["X_test"].copy()
    X["fraud_prob"] = prob.round(4)
    X["actual"] = y
    fnr = X[(X["actual"] == 1) & (X["fraud_prob"] < thr)].sort_values("fraud_prob")
    fpr = X[(X["actual"] == 0) & (X["fraud_prob"] >= thr)].sort_values("fraud_prob", ascending=False)
    st.markdown(f"**❌ Miss hue fraud (False Negatives): {len(fnr)}**")
    ui.table(fnr.head(50).round(2))
    st.markdown(f"**🚨 False alarms (False Positives): {len(fpr)}**")
    ui.table(fpr.head(50).round(2))
    ui.tip("Missed fraud ko dhyan se dekho: kya unme koi common pattern hai (jaise partial amount)? Wahi next feature idea hai.")

ui.next_step(8, "Agla step: Tuning & Cross-Validation")
