import os

import pandas as pd
import streamlit as st

from core import deploy, state, ui
from core import features as F

ui.setup("Model Deployment", "🚀", 11, "Model ko save karo, live prediction karo, API aur Docker banao")

ui.explain(
    "**Deployment = model ko kaam mein lagana.** 4 tareeqe:\n\n"
    "1. **Save:** model ek `.joblib` file mein save hota hai (feature list aur threshold ke saath)\n"
    "2. **Single predict:** ek transaction ka form bharo, fraud probability dekho\n"
    "3. **Batch predict:** CSV upload karo, saari transactions ka score paao\n"
    "4. **API / Docker:** doosre apps model ko call kar sakein (FastAPI)\n\n"
    "Training nahi ki? Koi problem nahi: 'Save' tab mein pehle se bani `.joblib` file upload kar sakte ho."
)

models = state.get("models")
sp = state.get("split")

# ---- bundle currently in use
bundle = st.session_state.get("bundle")

tabs = st.tabs(["💾 Save / Load model", "🧾 Single prediction", "📄 Batch prediction", "🔌 API (FastAPI)", "🐳 Docker & Hosting"])

with tabs[0]:
    if models and sp:
        names = list(models)
        name = st.selectbox("Kaun sa model?", names, index=names.index(state.get("best_model")) if state.get("best_model") in names else 0)
        thr = st.slider("Decision threshold", 0.01, 0.99, float(state.get("threshold", 0.5)), 0.01,
                        help="Evaluation page par jo chuna tha wahi default hai.")
        opts = state.get("feat_opts", F.DEFAULT_OPTS)
        if st.button("📦 Model bundle banao", type="primary"):
            bundle = deploy.build_bundle(name, models[name]["pipeline"], sp["features"], opts, thr,
                                         models[name]["metrics"], state.get("train_cfg"))
            st.session_state["bundle"] = bundle
            os.makedirs("models", exist_ok=True)
            try:
                with open("models/fraud_model.joblib", "wb") as f:
                    f.write(deploy.bundle_to_bytes(bundle))
                st.success("Bundle ban gaya aur `models/fraud_model.joblib` mein save hua (FastAPI isi ko use karti hai).")
            except OSError:
                st.success("Bundle ban gaya (server par file save nahi ho saki, download use karo).")
            state.put("best_model", name)
    else:
        st.info("Abhi train kiya hua model nahi hai. Neeche se purana bundle upload kar sakte ho.")

    up = st.file_uploader("Ya purana `.joblib` bundle upload karo", type=["joblib"])
    if up is not None and st.button("📤 Bundle load karo"):
        try:
            st.session_state["bundle"] = deploy.bundle_from_bytes(up.getvalue())
            st.success("Bundle load ho gaya.")
        except Exception as e:
            st.error(f"Load nahi hua: {e}")
    bundle = st.session_state.get("bundle")
    if bundle:
        st.markdown("---")
        st.markdown(f"**Active model:** {bundle['name']} · v{bundle['version']} · {bundle['created']}")
        c = st.columns(4)
        c[0].metric("Features", len(bundle["features"]))
        c[1].metric("Threshold", f"{bundle['threshold']:.2f}")
        c[2].metric("PR-AUC", f"{bundle['metrics'].get('pr_auc', float('nan')):.3f}")
        c[3].metric("Recall", f"{bundle['metrics'].get('recall', float('nan')):.3f}")
        st.download_button("⬇️ Model download (.joblib)", deploy.bundle_to_bytes(bundle), file_name="fraud_model.joblib")

if not bundle:
    for t in tabs[1:3]:
        with t:
            st.warning("Pehle '💾 Save / Load model' tab mein model bundle banao ya upload karo.")

with tabs[1]:
    if bundle:
        st.markdown("**Transaction ki details bharo:**")
        preset = st.radio("Example", ["Custom", "🚨 Suspicious TRANSFER", "✅ Normal PAYMENT"], horizontal=True)
        d = dict(type="TRANSFER", amount=181000.0, o=181000.0, n=0.0, od=0.0, nd=0.0) if preset.startswith("🚨") else \
            dict(type="PAYMENT", amount=1200.0, o=25000.0, n=23800.0, od=0.0, nd=0.0) if preset.startswith("✅") else \
            dict(type="CASH_OUT", amount=5000.0, o=20000.0, n=15000.0, od=10000.0, nd=15000.0)
        c1, c2, c3 = st.columns(3)
        ttype = c1.selectbox("Type", F.TYPES, index=F.TYPES.index(d["type"]), key=f"t_{preset}")
        amount = c2.number_input("Amount", 0.0, 1e10, d["amount"], 100.0, key=f"a_{preset}")
        step = c3.number_input("Step (ghanta)", 1, 10000, 100, key=f"s_{preset}")
        c1, c2 = st.columns(2)
        oo = c1.number_input("Sender balance PEHLE", 0.0, 1e10, d["o"], 100.0, key=f"oo_{preset}")
        no = c2.number_input("Sender balance BAAD MEIN", 0.0, 1e10, d["n"], 100.0, key=f"no_{preset}")
        c1, c2 = st.columns(2)
        od = c1.number_input("Receiver balance PEHLE", 0.0, 1e10, d["od"], 100.0, key=f"od_{preset}")
        nd = c2.number_input("Receiver balance BAAD MEIN", 0.0, 1e10, d["nd"], 100.0, key=f"nd_{preset}")
        merch = st.checkbox("Receiver merchant hai?", value=(ttype in ("PAYMENT", "DEBIT")))
        row = dict(step=step, type=ttype, amount=amount, oldbalanceOrg=oo, newbalanceOrig=no,
                   oldbalanceDest=od, newbalanceDest=nd, nameOrig="C0", nameDest="M0" if merch else "C0")
        if st.button("🔮 Predict karo", type="primary"):
            out = deploy.predict(bundle, pd.DataFrame([row])).iloc[0]
            p = float(out["fraud_probability"])
            color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[out["risk"]]
            a, b, c = st.columns(3)
            a.metric("Fraud probability", f"{p:.1%}")
            b.metric("Decision", out["decision"])
            c.metric("Risk level", f"{color} {out['risk']}")
            st.progress(min(p, 1.0))
            st.markdown("**Kyun? (red flags)**")
            for r in deploy.red_flags(row):
                st.markdown(f"- {r}")
            if out["decision"] == "FRAUD":
                st.error("🚨 Is transaction ko hold karke analyst/customer verification karo.")
            else:
                st.success("✅ Transaction allow kar sakte ho.")

with tabs[2]:
    if bundle:
        st.markdown("CSV mein ye columns chahiye: `type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest` "
                    "(optional: `step, nameOrig, nameDest`). `isFraud` ho to ignore hoga.")
        tmpl = pd.DataFrame([dict(step=1, type="TRANSFER", amount=181000, nameOrig="C1", oldbalanceOrg=181000, newbalanceOrig=0,
                                  nameDest="C2", oldbalanceDest=0, newbalanceDest=0)])
        st.download_button("⬇️ Template CSV", tmpl.to_csv(index=False), file_name="template.csv")
        f = st.file_uploader("Transactions CSV", type=["csv"], key="batch")
        if f is not None:
            raw = pd.read_csv(f)
            need = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            miss = [c for c in need if c not in raw.columns]
            if miss:
                st.error(f"Ye columns nahi mile: {', '.join(miss)}")
            else:
                with st.spinner("Predict ho raha hai…"):
                    res = deploy.predict(bundle, raw.drop(columns=["isFraud"], errors="ignore"))
                a, b, c = st.columns(3)
                a.metric("Rows", f"{len(res):,}")
                b.metric("Fraud flagged", f"{(res['decision'] == 'FRAUD').sum():,}")
                c.metric("Flag rate", f"{(res['decision'] == 'FRAUD').mean():.2%}")
                ui.table(res.sort_values("fraud_probability", ascending=False).head(500))
                ui.download_df(res, "predictions.csv", "⬇️ Predictions download")
                st.session_state["last_batch"] = res

with tabs[3]:
    st.markdown("**FastAPI service** doosre systems ko model use karne deti hai. File `api/main.py` project mein pehle se hai.")
    st.markdown("**1. Chalao** (model bundle pehle `models/fraud_model.joblib` mein save hona chahiye):")
    st.code("pip install fastapi uvicorn\nuvicorn api.main:app --reload --port 8000", language="bash")
    st.markdown("**2. Test karo:** browser mein `http://localhost:8000/docs` (Swagger UI) kholo, ya:")
    st.code(deploy.CURL_EXAMPLE, language="bash")
    st.markdown("**Endpoints:** `GET /health` · `GET /model-info` · `POST /predict` · `POST /predict_batch`")
    with st.expander("api/main.py ka code"):
        st.code(deploy.FASTAPI_CODE, language="python")

    st.markdown("**Live test** (API chal rahi ho to):")
    url = st.text_input("API URL", "http://localhost:8000")
    if st.button("🔌 /health check karo"):
        try:
            import requests
            st.json(requests.get(url.rstrip("/") + "/health", timeout=5).json())
        except Exception as e:
            st.error(f"API tak nahi pahunche: {e}")

with tabs[4]:
    st.markdown("### 🐳 Docker")
    a, b = st.columns(2)
    with a:
        st.markdown("**Dashboard (Streamlit)**")
        st.code(deploy.DOCKERFILE_APP, language="dockerfile")
        st.code("docker build -t fraud-app .\ndocker run -p 8501:8501 fraud-app", language="bash")
    with b:
        st.markdown("**Prediction API (FastAPI)**")
        st.code(deploy.DOCKERFILE_API, language="dockerfile")
        st.code("docker build -f Dockerfile.api -t fraud-api .\ndocker run -p 8000:8000 fraud-api", language="bash")
    st.markdown("### ☁️ Free hosting options")
    st.markdown(
        "- **Streamlit Community Cloud:** GitHub par code daalo → share.streamlit.io → *New app* → `app.py` chuno\n"
        "- **Hugging Face Spaces:** New Space → SDK *Streamlit* → files upload\n"
        "- **Render / Railway:** `Dockerfile.api` se API deploy\n"
        "- **AWS / Azure / GCP:** Docker image ko Cloud Run / App Service / ECS par chalao"
    )
    st.warning("Production mein: HTTPS, API authentication (API key/JWT), rate-limit, aur request logging zaroor lagao.")

ui.next_step(11, "Agla step: Monitoring & Prevention")
