import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import state, ui

ui.setup("Visualization", "📊", 3, "Basic se Advanced charts. Har chart ke neeche batata hai ki isse kya seekhna hai")
raw = state.require("raw_df", "Data Collection", "pages/1_Data_Collection.py")
df = state.get("clean_df") if state.get("clean_df") is not None else raw
CMAP = {"Normal": "#2E86AB", "Fraud": "#E4572E"}

ui.explain(
    "**Chart se pattern jaldi dikhta hai.** Neeche 3 level hain: **Basic** (ek column ko samjho), "
    "**Intermediate** (do cheezon ka rishta), **Advanced** (fraud ka asli pattern). Charts fast rakhne ke liye bade data se "
    "sample liya jata hai, lekin **saare fraud rows hamesha rakhe jate hain.**"
)
n = st.sidebar.slider("Chart sample size", 5000, 100000, 30000, step=5000)
d = ui.sample_for_plot(df, n).copy()
d["class"] = d["isFraud"].map({0: "Normal", 1: "Fraud"})
if {"newbalanceOrig", "oldbalanceOrg"} <= set(d.columns):
    d["balanceChangeOrig"] = d["newbalanceOrig"] - d["oldbalanceOrg"]
    d["errorBalanceOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]
if {"newbalanceDest", "oldbalanceDest"} <= set(d.columns):
    d["balanceChangeDest"] = d["newbalanceDest"] - d["oldbalanceDest"]
    d["errorBalanceDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]
if "step" in d.columns:
    d["hour_of_day"] = d["step"] % 24
    d["day"] = d["step"] // 24
num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c != "isFraud"]

lv = st.tabs(["🟢 Basic", "🟡 Intermediate", "🔴 Advanced", "🎨 Custom chart"])

# ---------------------------------------------------------------- BASIC
with lv[0]:
    st.markdown("#### 1. Histogram: ek column ka distribution")
    col = st.selectbox("Column", num_cols, index=num_cols.index("amount") if "amount" in num_cols else 0, key="b1")
    logx = st.checkbox("Log scale (skewed data ke liye)", value=True, key="b1l")
    ui.plot(px.histogram(d, x=col, nbins=60, log_y=logx, color="class", color_discrete_map=CMAP, barmode="overlay", opacity=0.7))
    st.caption("Seekho: bahut saari values chhoti, kuch bahut badi (long tail). Fraud ka distribution normal se alag dikhe to wo column useful hai.")

    a, b = st.columns(2)
    with a:
        st.markdown("#### 2. Bar: transaction types")
        vc = df["type"].value_counts().reset_index()
        vc.columns = ["type", "count"]
        ui.plot(px.bar(vc, x="type", y="count", color="type"))
    with b:
        st.markdown("#### 3. Pie: fraud vs normal")
        ui.plot(px.pie(d, names="class", color="class", color_discrete_map=CMAP, hole=0.4))

    st.markdown("#### 4. Box plot: fraud vs normal")
    col2 = st.selectbox("Column", num_cols, index=num_cols.index("oldbalanceOrg") if "oldbalanceOrg" in num_cols else 0, key="b4")
    ui.plot(px.box(d, x="class", y=col2, color="class", color_discrete_map=CMAP, log_y=st.checkbox("Log y", True, key="b4l")))
    st.caption("Seekho: dabba (box) kahan hai, median kahan hai. Dono classes ke box alag hon to model ko asaani hogi.")

# ---------------------------------------------------------------- INTERMEDIATE
with lv[1]:
    st.markdown("#### 5. Fraud rate per transaction type")
    g = df.groupby("type")["isFraud"].agg(["mean", "sum", "size"]).reset_index()
    g["fraud_%"] = g["mean"] * 100
    ui.plot(px.bar(g, x="type", y="fraud_%", text=g["sum"].astype(str) + " fraud", color_discrete_sequence=["#E4572E"]))
    st.caption("Seekho: fraud kuch hi types mein hota hai. Ye model ka sabse bada hint hai.")

    st.markdown("#### 6. Correlation heatmap")
    sel = st.multiselect("Columns", num_cols, default=[c for c in num_cols if c not in ("day",)][:10], key="corr_cols")
    if len(sel) >= 2:
        from core.plots import corr_fig
        ui.plot(corr_fig(d.assign(isFraud=d["isFraud"]), sel + ["isFraud"]))
        st.caption("Seekho: lal = saath badhte hain, neela = ulte. Do features ka corr ~1 ho to dono same info dete hain (multicollinearity).")

    if "step" in d.columns:
        st.markdown("#### 7. Time series: transactions / fraud over time")
        tm = d.copy()
        tm["day"] = tm["step"] // 24
        ts = df.assign(day=df["step"] // 24).groupby("day").agg(transactions=("isFraud", "size"), fraud=("isFraud", "sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["day"], y=ts["transactions"], name="Transactions", line=dict(color="#2E86AB")))
        fig.add_trace(go.Scatter(x=ts["day"], y=ts["fraud"], name="Fraud", yaxis="y2", line=dict(color="#E4572E")))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Fraud"), xaxis_title="Day", yaxis_title="Transactions")
        ui.plot(fig)
        st.caption("Seekho: fraud kisi khaas din spike karta hai to wahan campaign/attack ho sakta hai.")

    st.markdown("#### 8. Violin: amount distribution by type & class")
    ui.plot(px.violin(d, x="type", y="amount", color="class", color_discrete_map=CMAP, log_y=True, box=True))

# ---------------------------------------------------------------- ADVANCED
with lv[2]:
    if "errorBalanceOrig" in d.columns:
        st.markdown("#### 9. Balance mismatch scatter (fraud ka signature)")
        ui.plot(px.scatter(d.sample(min(len(d), 15000), random_state=1), x="errorBalanceOrig", y="errorBalanceDest",
                           color="class", color_discrete_map=CMAP, opacity=0.6))
        st.caption("Seekho: sahi transaction mein dono error ~0 hote hain. Fraud rows ek alag jagah cluster banati hain.")

    if "hour_of_day" in d.columns:
        st.markdown("#### 10. Heatmap: din ke ghante × type par fraud count")
        hm = df.assign(hour=df["step"] % 24).query("isFraud == 1").pivot_table(
            index="type", columns="hour", values="isFraud", aggfunc="sum", fill_value=0)
        if hm.empty:
            st.info("Fraud rows nahi hain.")
        else:
            ui.plot(px.imshow(hm, aspect="auto", color_continuous_scale="Reds", labels=dict(color="Fraud")))

    st.markdown("#### 11. Sankey: type → fraud / normal")
    s = df.groupby(["type", "isFraud"]).size().reset_index(name="n")
    types = list(s["type"].unique())
    labels = types + ["Normal", "Fraud"]
    src = [types.index(t) for t in s["type"]]
    tgt = [len(types) + int(f) for f in s["isFraud"]]
    ui.plot(go.Figure(go.Sankey(node=dict(label=labels, pad=18), link=dict(source=src, target=tgt, value=s["n"].tolist()))))

    if len(num_cols) >= 3:
        st.markdown("#### 12. Scatter matrix")
        pm = st.multiselect("3-5 columns", num_cols, default=[c for c in ["amount", "oldbalanceOrg", "newbalanceOrig"] if c in num_cols], key="pm")
        if len(pm) >= 2:
            ui.plot(px.scatter_matrix(d.sample(min(len(d), 4000), random_state=2), dimensions=pm, color="class",
                                      color_discrete_map=CMAP, opacity=0.5))

    st.markdown("#### 13. Fraud rate by amount bucket")
    try:
        b = pd.qcut(df["amount"], 10, duplicates="drop")
        r = df.groupby(b, observed=True)["isFraud"].mean().mul(100).reset_index()
        r["amount"] = r["amount"].astype(str)
        ui.plot(px.bar(r, x="amount", y="isFraud", labels={"isFraud": "Fraud %"}, color_discrete_sequence=["#E4572E"]))
    except Exception:
        st.info("Is column par buckets nahi ban paye.")

# ---------------------------------------------------------------- CUSTOM
with lv[3]:
    st.markdown("Apna chart khud banao.")
    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Chart type", ["Scatter", "Histogram", "Box", "Bar (mean)", "Line (mean)"])
    x = c2.selectbox("X", d.columns, index=list(d.columns).index("type") if "type" in d.columns else 0)
    y = c3.selectbox("Y", [None] + num_cols)
    colr = c4.checkbox("Fraud/Normal rang", True)
    kw = dict(color="class", color_discrete_map=CMAP) if colr else {}
    try:
        if kind == "Scatter" and y:
            fig = px.scatter(d, x=x, y=y, opacity=0.6, **kw)
        elif kind == "Histogram":
            fig = px.histogram(d, x=x, **kw)
        elif kind == "Box" and y:
            fig = px.box(d, x=x, y=y, **kw)
        elif kind == "Bar (mean)" and y:
            fig = px.bar(d.groupby(x)[y].mean().reset_index(), x=x, y=y)
        elif kind == "Line (mean)" and y:
            fig = px.line(d.groupby(x)[y].mean().reset_index().sort_values(x), x=x, y=y)
        else:
            fig = None
            st.info("Is chart ke liye Y column bhi chuno.")
        if fig is not None:
            ui.plot(fig)
    except Exception as e:
        st.error(f"Chart nahi ban saka: {e}")

ui.next_step(3, "Agla step: Preprocessing (data saaf karo)")
