import streamlit as st

from core import loaders, state, ui

ui.setup("Data Collection", "📥", 1, "Dataset load karo: apni file / Google Drive / Kaggle API / demo")

ui.explain(
    "**Is step mein kya hota hai?** Model ko sikhane ke liye data chahiye. Yahan 4 tarike hain data lane ke. "
    "Data aane ke baad app check karti hai ki zaroori columns hain ya nahi, aur phir tum use pipeline mein save karte ho.\n\n"
    "**Zaroori columns:** `type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud`\n\n"
    "**Optional:** `step, nameOrig, nameDest, isFlaggedFraud`",
    expanded=True,
)

tab_up, tab_drive, tab_kaggle, tab_demo = st.tabs(
    ["📁 File upload", "☁️ Google Drive link", "🔑 Kaggle API", "⚡ Demo data"])

if "pending_df" not in st.session_state:
    st.session_state["pending_df"] = None
    st.session_state["pending_src"] = ""


def stage(df, src):
    st.session_state["pending_df"] = df
    st.session_state["pending_src"] = src


with tab_up:
    st.markdown("CSV / Excel / Parquet / ZIP (andar CSV) upload karo. **Drive se file download karke yahan daal sakte ho.**")
    f = st.file_uploader("File chuno", type=["csv", "xlsx", "xls", "parquet", "zip", "gz"])
    if f is not None and st.button("📂 File padho", key="read_file"):
        try:
            with st.spinner("File padh rahe hain…"):
                stage(loaders.read_any(f, f.name), f"Upload: {f.name}")
            st.success("File padh li gayi. Neeche preview dekho.")
        except Exception as e:
            st.error(f"File nahi padhi ja saki: {e}")

with tab_drive:
    st.markdown(
        "**Steps:** (1) Drive mein file par right-click → *Share* → **Anyone with the link** → *Copy link*. "
        "(2) Link yahan paste karo."
    )
    url = st.text_input("Google Drive share link", placeholder="https://drive.google.com/file/d/<FILE_ID>/view")
    if st.button("☁️ Drive se download karo", key="drive_btn"):
        try:
            with st.spinner("Drive se download ho raha hai…"):
                stage(loaders.load_gdrive(url), "Google Drive")
            st.success("Download ho gaya.")
        except Exception as e:
            st.error(f"Drive se nahi aa saka: {e}")
    st.caption("Tip: bahut badi files par Google 'virus scan' page deta hai. Us case mein file ko local download karke 'File upload' use karo.")

with tab_kaggle:
    st.markdown("Kaggle ke liye **API token** chahiye: kaggle.com → *Settings* → *API* → *Create New Token* (`kaggle.json` milegi).")
    st.markdown("**Terminal command (agar khud chalana ho):**")
    slug = st.text_input("Dataset slug", value=loaders.KAGGLE_SLUG, help="kaggle.com/datasets/<owner>/<dataset>")
    st.code(loaders.kaggle_command(slug), language="bash")

    kj = st.file_uploader("kaggle.json upload karo (credentials auto-fill honge)", type=["json"], key="kj")
    if kj is not None:
        try:
            u, k = loaders.parse_kaggle_json(kj.getvalue())
            st.session_state["k_user"], st.session_state["k_key"] = u, k
            st.success(f"Credentials mil gaye: {u}")
        except Exception:
            st.error("kaggle.json sahi format mein nahi hai.")
    c1, c2 = st.columns(2)
    user = c1.text_input("Kaggle username", key="k_user")
    key = c2.text_input("Kaggle API key", type="password", key="k_key")
    st.caption("🔒 Credentials sirf is session ki memory mein rehte hain, disk par save nahi hote.")
    if st.button("🔑 Kaggle se download karo", key="kaggle_btn"):
        try:
            with st.spinner("Kaggle se download ho raha hai (thoda time lag sakta hai)…"):
                df, files = loaders.load_kaggle(slug, user, key)
            stage(df, f"Kaggle: {slug}")
            st.success(f"Download ho gaya. Files: {', '.join(files)}")
        except Exception as e:
            st.error(f"Kaggle error: {e}")

with tab_demo:
    st.markdown("Bina kuch download kiye app try karna hai? **Synthetic data** (PaySim jaisa pattern: fraud sirf TRANSFER/CASH_OUT mein).")
    n = st.slider("Rows", 5000, 100000, 20000, step=5000)
    fr = st.slider("Fraud %", 0.5, 10.0, 2.0, step=0.5) / 100
    if st.button("⚡ Demo data banao", key="demo_btn"):
        stage(loaders.make_demo_data(n, fr), "Demo (synthetic)")
        st.success("Demo data ban gaya.")

# ------------------------------------------------------------------ review & save
df = st.session_state["pending_df"]
st.divider()
if df is None:
    cur = state.get("raw_df")
    if cur is not None:
        st.success(f"✅ Pipeline mein data already hai: **{state.get('data_source', '')}** · {cur.shape[0]:,} rows × {cur.shape[1]} columns")
        ui.table(cur.head(10))
        ui.next_step(1, "Agla step: EDA (data samjho)")
    else:
        st.info("Upar se koi ek tarika chuno.")
    st.stop()

st.subheader("🔎 Preview & validation")
st.caption(f"Source: {st.session_state['pending_src']}")
a, b, c = st.columns(3)
a.metric("Rows", f"{df.shape[0]:,}")
b.metric("Columns", df.shape[1])
c.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
ui.table(df.head(10))

miss_req, miss_opt = loaders.validate_schema(df)
if miss_req:
    st.error(f"Ye zaroori columns nahi mile: **{', '.join(miss_req)}**. Neeche apne columns se match karo.")
    mapping = {}
    opts = [""] + list(df.columns)
    cols = st.columns(3)
    for i, m in enumerate(miss_req):
        mapping[m] = cols[i % 3].selectbox(f"'{m}' = ", opts, key=f"map_{m}")
    if all(mapping.values()):
        df = loaders.apply_mapping(df, mapping)
        miss_req, miss_opt = loaders.validate_schema(df)
        if not miss_req:
            st.success("Columns match ho gaye ✅")
else:
    st.success("✅ Saare zaroori columns mil gaye.")
if miss_opt and not miss_req:
    st.caption(f"Optional columns nahi hain (theek hai): {', '.join(miss_opt)}")

if not miss_req:
    if df["isFraud"].nunique() < 2:
        st.warning("⚠️ `isFraud` mein sirf ek hi value hai. Model train nahi ho payega.")
    else:
        st.caption(f"Fraud rows: **{int(df['isFraud'].sum()):,}** ({df['isFraud'].mean() * 100:.3f}%)")
    if st.button("✅ Is data ko pipeline mein save karo", type="primary"):
        state.reset_from("raw_df")
        state.put("raw_df", df.reset_index(drop=True))
        state.put("data_source", st.session_state["pending_src"])
        st.session_state["pending_df"] = None
        st.success("Data save ho gaya! Ab agle step par jao.")
        st.balloons()
        st.rerun()
