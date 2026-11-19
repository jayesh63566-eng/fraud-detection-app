import pandas as pd
import plotly.express as px
import streamlit as st

from core import features as F
from core import state, ui

ui.setup("Feature Engineering", "🛠️", 5, "Naye, smart columns banao jo fraud ka pattern pakadte hain")
clean = state.require("clean_df", "Preprocessing", "pages/4_Preprocessing.py")

ui.explain(
    "**Feature = model ko diya gaya ek column.** Raw columns se naye, zyada meaningful columns banana hi **feature engineering** hai.\n\n"
    "Example: sender ka `oldbalance − amount` agar `newbalance` ke barabar nahi hai, to kuch gadbad hai. "
    "Ye 'errorBalance' feature fraud pakadne mein bahut powerful hota hai.\n\n"
    "Neeche checkboxes se chuno kaun se features chahiye. Har ek ke saath chhoti explanation hai."
)

LABELS = {
    "balance_change": "Balance change (new − old)",
    "error_balance": "Error balance (accounting mismatch) ⭐",
    "flags": "Red-flag columns (account khaali, poora balance, dest unchanged, high amount) ⭐",
    "time_parts": "Time parts (hour, day, night)",
    "log_amount": "Log of amount",
    "merchant_flag": "Merchant receiver flag",
    "txn_counts": "Sender/receiver transaction count (history chahiye)",
    "type_onehot": "Type ko one-hot encode karo",
    "drop_ids": "Account IDs hatao (nameOrig / nameDest)",
    "drop_flagged": "isFlaggedFraud hatao",
}
opts_prev = state.get("feat_opts", F.DEFAULT_OPTS)

c1, c2 = st.columns(2)
opts = {}
for i, (k, lab) in enumerate(LABELS.items()):
    with (c1 if i % 2 == 0 else c2):
        opts[k] = st.checkbox(lab, value=opts_prev.get(k, F.DEFAULT_OPTS[k]), help=F.FEATURE_HELP[k], key=f"fe_{k}")

if opts["txn_counts"]:
    st.warning("⚠️ Transaction-count feature live prediction mein sirf batch mein kaam karega (history chahiye). Single prediction mein 1 maana jayega.")
if not opts["type_onehot"]:
    st.warning("⚠️ Type encode nahi karoge to `type` model mein nahi jayega, jabki wo sabse zaroori feature hai.")

if st.button("🛠️ Features banao", type="primary"):
    feat, created = F.engineer(clean, opts)
    state.reset_from("feat_df")
    state.put("feat_df", feat)
    state.put("feat_opts", opts)
    state.put("feat_created", created)
    st.rerun()

feat = state.get("feat_df")
if feat is None:
    st.info("Upar se features chuno aur 'Features banao' dabao.")
    st.stop()

created = state.get("feat_created", [])
st.success(f"✅ {len(created)} naye features bane. Ab total columns: {feat.shape[1]}")
t1, t2, t3 = st.tabs(["🆕 Naye features", "📊 Fraud se rishta", "📋 Poora table"])

with t1:
    ui.table(feat[created].head(10).round(2) if created else pd.DataFrame())
    st.caption("Naye columns: " + ", ".join(f"`{c}`" for c in created))

with t2:
    st.markdown("**Kaun sa naya feature fraud se sabse zyada juda hai?** (|correlation| with isFraud)")
    num = [c for c in created if pd.api.types.is_numeric_dtype(feat[c])]
    if num:
        corr = feat[num + ["isFraud"]].corr()["isFraud"].drop("isFraud").dropna()
        cdf = corr.reindex(corr.abs().sort_values(ascending=False).index).reset_index()
        cdf.columns = ["feature", "corr_with_fraud"]
        ui.plot(px.bar(cdf.iloc[::-1], x="corr_with_fraud", y="feature", orientation="h",
                       color="corr_with_fraud", color_continuous_scale="RdBu_r", range_color=[-1, 1]))
        pick = st.selectbox("Ek feature ka fraud vs normal comparison", num)
        sm = ui.sample_for_plot(feat, 20000)
        ui.plot(px.box(sm, x=sm["isFraud"].astype(str), y=pick, labels={"x": "isFraud"}, color=sm["isFraud"].astype(str),
                       color_discrete_map={"0": "#2E86AB", "1": "#E4572E"}))

with t3:
    ui.table(feat.head(100).round(2), height=420)
    ui.download_df(feat, "features.csv", "⬇️ Feature data download")

ui.next_step(5, "Agla step: Feature Selection")
