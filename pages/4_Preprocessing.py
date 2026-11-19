import pandas as pd
import plotly.express as px
import streamlit as st

from core import cleaning, state, ui

ui.setup("Preprocessing (Data Cleaning)", "🧹", 4, "Missing values, duplicates, outliers, data types")
raw = state.require("raw_df", "Data Collection", "pages/1_Data_Collection.py")

ui.explain(
    "**Preprocessing = data ko model ke layak banana.** Is page par 5 cheezein hain: data types, missing values, duplicates, "
    "outliers. Har section mein pehle problem dikhti hai, phir tum choose karte ho kya karna hai. "
    "Last mein **'Apply'** dabate ho. Original data safe rehta hai, tum kabhi bhi dobara apply kar sakte ho.\n\n"
    "⚠️ **Scaling aur SMOTE yahan nahi hote.** Wo sirf *train* data par hone chahiye (warna test ka data leak ho jata hai), "
    "isliye wo *Model Training* page par hote hain."
)

num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("isFraud", "isFlaggedFraud", "step")]
tabs = st.tabs(["1️⃣ Data types", "2️⃣ Missing", "3️⃣ Duplicates", "4️⃣ Outliers", "5️⃣ Scaling (demo)", "✅ Apply"])

with tabs[0]:
    ui.table(pd.DataFrame({"column": raw.columns, "dtype": raw.dtypes.astype(str).values, "example": raw.iloc[0].astype(str).values}))
    cfg_type_cat = st.checkbox("`type` ko category dtype banao (memory bachti hai)", value=True)

with tabs[1]:
    mr = cleaning.missing_report(raw)
    if mr["missing"].sum() == 0:
        st.success("✅ Is data mein koi missing value nahi hai. Kuch karne ki zarurat nahi.")
        miss_mode, num_s, cat_s = "fill", "median", "most_frequent"
    else:
        ui.table(mr[mr["missing"] > 0])
        miss_mode = st.radio("Kya karein?", ["fill", "drop"], format_func=lambda x: {"fill": "Fill karo (recommended)", "drop": "Rows hata do"}[x], horizontal=True)
        num_s = st.selectbox("Numeric fill", ["median", "mean", "zero"], help="Median outliers se kam prabhavit hota hai")
        cat_s = st.selectbox("Text fill", ["most_frequent", "unknown"])

with tabs[2]:
    nd = int(raw.duplicated().sum())
    st.metric("Duplicate rows", f"{nd:,}")
    drop_dup = st.checkbox("Duplicates hata do", value=True)
    if nd == 0:
        st.success("✅ Koi duplicate nahi.")

with tabs[3]:
    st.markdown("**Outlier kya hai?** Bahut bada/chhota value. Fraud detection mein ye **useful signal** ho sakte hain, isliye default mein hum unhe **rakhte hain**.")
    c1, c2 = st.columns(2)
    method = c1.selectbox("Detection method", ["IQR", "Z-score", "Percentile"])
    out_cols = c2.multiselect("Kin columns par?", num_cols, default=[c for c in ["amount"] if c in num_cols])
    if out_cols:
        rep = cleaning.outlier_report(raw, out_cols, method)
        ui.table(rep)
        st.caption("'fraud_% inside outliers' = outliers mein fraud ka share. Ye overall fraud% se bahut zyada ho to outliers hatana galat hoga.")
        col = st.selectbox("Distribution dekho", out_cols)
        sm = ui.sample_for_plot(raw, 20000)
        ui.plot(px.box(sm, x=sm["isFraud"].astype(str), y=col, log_y=True, labels={"x": "isFraud"}))
    action = st.radio("Action", ["keep", "cap", "remove"],
                      format_func=lambda x: {"keep": "Rakho (recommended)", "cap": "Cap karo (limit par kaat do)", "remove": "Rows hata do"}[x], horizontal=True)
    if action == "remove":
        st.warning("⚠️ Remove karne se **fraud rows bhi ud sakti hain.** Careful raho.")

with tabs[4]:
    st.markdown("**Scaling kyun?** Amount lakhon mein, `isNight` 0/1 mein. KNN, SVC, Logistic jaise models ko scale chahiye. Tree models ko nahi.")
    show = st.multiselect("Demo columns", num_cols, default=[c for c in ["amount", "oldbalanceOrg"] if c in num_cols])
    if show:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        s = raw[show].sample(min(len(raw), 5000), random_state=0)
        std = pd.DataFrame(StandardScaler().fit_transform(s), columns=[f"{c} (standard)" for c in show])
        mm = pd.DataFrame(MinMaxScaler().fit_transform(s), columns=[f"{c} (min-max)" for c in show])
        a, b, c = st.columns(3)
        a.markdown("**Original**"); a.dataframe(s.describe().loc[["mean", "std", "min", "max"]].round(1))
        b.markdown("**StandardScaler** (mean 0, std 1)"); b.dataframe(std.describe().loc[["mean", "std", "min", "max"]].round(2))
        c.markdown("**MinMaxScaler** (0 se 1)"); c.dataframe(mm.describe().loc[["mean", "std", "min", "max"]].round(2))
    st.info("Asli scaling model training ke time pipeline ke andar hogi, sirf un models ke liye jinko zarurat hai.")

with tabs[5]:
    st.markdown("Upar ki settings se data saaf karo:")
    if st.button("🧹 Preprocessing apply karo", type="primary"):
        d, log = raw.copy(), []
        n0 = len(d)
        if cfg_type_cat and "type" in d.columns:
            d["type"] = d["type"].astype(str)
            log.append("`type` ko text (string) rakha taaki aage encoding aasaan ho.")
        d, l = cleaning.impute(d, num_s, cat_s, drop_rows=(miss_mode == "drop"))
        log += l
        if drop_dup:
            n1 = len(d)
            d = d.drop_duplicates()
            log.append(f"{n1 - len(d)} duplicate rows hataye.")
        d, l = cleaning.handle_outliers(d, out_cols, method, action)
        log += l
        d = d.reset_index(drop=True)
        state.reset_from("clean_df")
        state.put("clean_df", d)
        state.put("prep_log", log)
        state.put("prep_stats", dict(before=n0, after=len(d), fraud_before=int(raw["isFraud"].sum()), fraud_after=int(d["isFraud"].sum())))
        st.rerun()

    cd = state.get("clean_df")
    if cd is not None:
        s = state.get("prep_stats", {})
        a, b, c = st.columns(3)
        a.metric("Rows (pehle → baad)", f"{s.get('before', 0):,} → {s.get('after', 0):,}")
        b.metric("Fraud rows", f"{s.get('fraud_before', 0):,} → {s.get('fraud_after', 0):,}")
        c.metric("Missing cells", int(cd.isnull().sum().sum()))
        st.markdown("**Kya kiya gaya:**")
        for l in state.get("prep_log", []):
            st.markdown(f"- {l}")
        st.success("✅ Saaf data pipeline mein save ho gaya.")
        ui.download_df(cd, "clean_data.csv", "⬇️ Clean data download")
        ui.next_step(4, "Agla step: Feature Engineering")
    else:
        st.info("Abhi apply nahi kiya.")
