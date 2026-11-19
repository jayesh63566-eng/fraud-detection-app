import pandas as pd
import plotly.express as px
import streamlit as st

from core import state, ui

ui.setup("Data Exploration (EDA)", "🔍", 2, "Data ko samjho, kuch badalne se pehle")
df = state.require("raw_df", "Data Collection", "pages/1_Data_Collection.py")

ui.explain(
    "**EDA = Exploratory Data Analysis.** Model banane se pehle data ko dekhna: kitni rows hain, kaun si values missing hain, "
    "fraud kitna hai, numbers ka range kya hai. Isse pata chalta hai ki aage kya saaf karna padega."
)

t = st.tabs(["📋 Overview", "👀 Head / Tail", "❓ Missing", "📐 Describe", "🔢 Unique values", "🎯 Target (fraud)", "🏷️ Type"])

with t[0]:
    c = st.columns(4)
    c[0].metric("Rows", f"{df.shape[0]:,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Duplicates", f"{df.duplicated().sum():,}")
    c[3].metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
    info = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
                         "non-null": df.notnull().sum().values, "unique": df.nunique().values})
    ui.table(info)

with t[1]:
    n = st.slider("Kitni rows?", 3, 50, 5)
    st.markdown("**Pehli rows (head)**")
    ui.table(df.head(n))
    st.markdown("**Aakhri rows (tail)**")
    ui.table(df.tail(n))

with t[2]:
    from core.cleaning import missing_report
    mr = missing_report(df)
    if mr["missing"].sum() == 0:
        st.success("✅ Koi missing value nahi hai.")
    else:
        ui.table(mr)
        ui.plot(px.bar(mr[mr["missing"] > 0], x="column", y="missing_%"))

with t[3]:
    st.markdown("**Numeric columns**")
    ui.table(df.describe().T.reset_index().rename(columns={"index": "column"}).round(2))
    cat = df.select_dtypes(exclude="number")
    if not cat.empty:
        st.markdown("**Text columns**")
        ui.table(cat.describe().T.reset_index().rename(columns={"index": "column"}))
    ui.tip("`amount` aur balances mein mean ≫ median ho to data **right-skewed** hai (kuch bahut bade values). Isiliye aage log-transform use karenge.")

with t[4]:
    u = df.nunique().sort_values(ascending=False).reset_index()
    u.columns = ["column", "unique values"]
    ui.table(u)
    ui.tip("nameOrig / nameDest mein lagbhag har row alag hoti hai (high cardinality). Ye IDs hain, model ke liye bekaar, isliye feature engineering mein hata denge.")

with t[5]:
    vc = df["isFraud"].value_counts().rename({0: "Normal", 1: "Fraud"}).reset_index()
    vc.columns = ["class", "count"]
    a, b = st.columns(2)
    with a:
        ui.plot(px.pie(vc, names="class", values="count", color="class",
                       color_discrete_map={"Normal": "#2E86AB", "Fraud": "#E4572E"}, hole=0.45))
    with b:
        fr = df["isFraud"].mean() * 100
        st.metric("Fraud %", f"{fr:.3f}%")
        if fr > 0:
            st.metric("Imbalance", f"1 fraud : {int((100 - fr) / fr):,} normal")
        st.warning("Data **bahut imbalanced** hai. Agar model sab ko 'normal' bol de to bhi accuracy ~"
                   f"{100 - fr:.1f}% aayegi! Isliye Precision, Recall aur PR-AUC dekhenge, aur imbalance handle karenge.")
    if "isFlaggedFraud" in df.columns:
        st.markdown("**isFlaggedFraud (purane rule-system ka flag)**")
        ui.table(pd.crosstab(df["isFraud"], df["isFlaggedFraud"]).reset_index())
        ui.tip("Agar flagged bahut kam hai jabki fraud zyada, to purana rule-system zyada fraud miss karta hai. Yahi ML ki zarurat hai.")

with t[6]:
    g = df.groupby("type")["isFraud"].agg(count="size", fraud="sum", fraud_pct=lambda s: s.mean() * 100).reset_index()
    a, b = st.columns(2)
    with a:
        ui.plot(px.bar(g, x="type", y="count", title="Transactions per type"))
    with b:
        ui.plot(px.bar(g, x="type", y="fraud_pct", title="Fraud % per type", color_discrete_sequence=["#E4572E"]))
    ui.table(g.round(3))
    hot = g[g["fraud"] > 0]["type"].tolist()
    if hot:
        st.success(f"Fraud sirf in types mein mila: **{', '.join(hot)}**")

ui.next_step(2, "Agla step: Visualization (charts)")
