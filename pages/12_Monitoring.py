from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from core import monitor, state, ui

ui.setup("Monitoring & Prevention", "🛡️", 12, "Production mein model theek chal raha hai? Fraud kaise rokein?")

ui.explain(
    "**Model deploy karke kaam khatam nahi hota.** Fraudsters apne tareeke badalte rehte hain, to model dheere-dheere kharab hota hai "
    "(**data drift / concept drift**). Is page par:\n\n"
    "1. **Performance log:** model ke scores waqt ke saath track karo\n"
    "2. **Drift check (PSI):** naya data purane training data se kitna alag hai\n"
    "3. **Retrain trigger:** kab model dobara train karna hai\n"
    "4. **Prevention:** infrastructure update mein kaun si suraksha lagani chahiye\n"
    "5. **Success kaise napein:** actions kaam kar rahe hain ya nahi"
)

tabs = st.tabs(["📈 Performance log", "🌊 Drift check", "🔁 Retrain trigger", "🛡️ Prevention (Q7)", "✅ Success measure (Q8)"])
hist = st.session_state.setdefault("perf_log", [])

with tabs[0]:
    models, sp = state.get("models"), state.get("split")
    if models and sp:
        nm = state.get("best_model") or list(models)[0]
        st.markdown(f"Current model: **{nm}**")
        if st.button("➕ Current metrics log mein jodo"):
            m = models[nm]["metrics"]
            hist.append(dict(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"), model=nm, pr_auc=m["pr_auc"],
                             recall=m["recall"], precision=m["precision"], f1=m["f1"]))
            st.rerun()
    else:
        st.info("Model train karne ke baad yahan se metrics log kar sakte ho. Neeche CSV upload bhi kar sakte ho.")
    up = st.file_uploader("Purana log CSV (timestamp, model, pr_auc, recall, precision, f1)", type=["csv"])
    if up is not None and st.button("📤 Log load karo"):
        st.session_state["perf_log"] = pd.read_csv(up).to_dict("records")
        st.rerun()
    if hist:
        h = pd.DataFrame(hist)
        ui.table(h.round(4))
        ui.plot(px.line(h.melt(id_vars=["timestamp", "model"], value_vars=["pr_auc", "recall", "precision", "f1"]),
                        x="timestamp", y="value", color="variable", markers=True))
        ui.download_df(h, "model_performance_log.csv", "⬇️ Log download")
        st.caption("Real deployment mein ye log roz/har hafte naye *labelled* transactions par bharta hai.")

with tabs[1]:
    sp = state.get("split")
    if not sp:
        st.warning("Drift check ke liye pehle model training (split) chahiye.")
    else:
        feats = sp["features"]
        mode = st.radio("Naya data kahan se?", ["🧪 Simulate drift (demo)", "📁 Apna naya CSV"], horizontal=True)
        new = None
        if mode.startswith("🧪"):
            s = st.slider("Drift strength", 0.0, 2.0, 0.5, 0.1)
            new = monitor.simulate_drift(sp["X_test"], feats, s)
        else:
            f = st.file_uploader("Naya data (engineered features wale columns, ya raw CSV)", type=["csv"], key="drift_up")
            if f is not None:
                raw = pd.read_csv(f)
                if all(c in raw.columns for c in feats):
                    new = raw
                else:
                    from core.features import engineer
                    from core.deploy import prepare_raw
                    new = engineer(prepare_raw(raw), state.get("feat_opts"))[0]
        if new is not None:
            rep = monitor.drift_report(sp["X_train"], new, feats)
            state.put("worst_psi", float(rep["PSI"].max()) if len(rep) else 0.0)
            ui.table(rep)
            ui.plot(px.bar(rep.head(15).iloc[::-1], x="PSI", y="feature", orientation="h", color="status",
                           color_discrete_map={"Stable ✅": "#2E86AB", "Moderate ⚠️": "#F4A261", "High 🚨": "#E4572E"}))
            st.caption("**PSI:** < 0.1 stable · 0.1–0.25 thoda badlav · > 0.25 bada badlav (retrain sochna chahiye).")
            if len(rep):
                top = st.selectbox("Distribution compare karo", rep["feature"])
                cmp = pd.concat([sp["X_train"][[top]].assign(data="Training").sample(min(5000, len(sp["X_train"])), random_state=0),
                                 new[[top]].assign(data="New").sample(min(5000, len(new)), random_state=0)])
                ui.plot(px.histogram(cmp, x=top, color="data", barmode="overlay", opacity=0.6, nbins=50))

with tabs[2]:
    st.markdown("**Retrain ke rules set karo:**")
    c1, c2, c3 = st.columns(3)
    min_rec = c1.number_input("Min Recall", 0.0, 1.0, 0.80, 0.01)
    min_pr = c2.number_input("Min PR-AUC", 0.0, 1.0, 0.60, 0.01)
    max_psi = c3.number_input("Max PSI", 0.0, 2.0, 0.25, 0.01)
    h = pd.DataFrame(hist)
    go_, why = monitor.retrain_decision(h, min_rec, min_pr, max_psi, state.get("worst_psi", 0.0))
    if not hist and not state.get("worst_psi"):
        st.info("Pehle performance log ya drift check chalao.")
    elif go_:
        st.error("🔁 **Retrain karna chahiye.** Wajah:")
        for r in why:
            st.markdown(f"- {r}")
    else:
        st.success("✅ Abhi retrain ki zarurat nahi. Sab thresholds ke andar hai.")
    st.markdown("**Retraining schedule suggestion:** har mahine ya jab trigger ho · naye labelled fraud cases add karo · "
                "pehle *shadow mode* mein naya model chalao, purane se compare karo, phir hi replace karo.")

with tabs[3]:
    st.markdown("### Infrastructure update ke time kaun si prevention lagayein?")
    st.markdown(
        "**A. Real-time controls**\n"
        "- Model ka fraud score har TRANSFER / CASH_OUT par real-time lagao; high risk par **hold + OTP / biometric** verification\n"
        "- **Velocity limits:** ek account se thode time mein bahut transfers rokna\n"
        "- Naye / pehli baar ke receiver par **cooling-off period** aur lower limit\n"
        "- Poora balance ek baar mein nikalne par extra confirmation\n\n"
        "**B. Account safety**\n"
        "- **Multi-factor authentication (MFA)** aur device fingerprinting; naye device se login par alert\n"
        "- Password reset ya SIM-swap ke baad kuch ghanton ke liye limits kam\n"
        "- Customer ko instant SMS/push alert har bade transaction par\n\n"
        "**C. Data & model**\n"
        "- Receiver balance update aur transaction logs **consistently** record karo (error-balance features inpe nirbhar hain)\n"
        "- Feature store: sender/receiver ki history (velocity, naya account) real-time uplabdh ho\n"
        "- Analysts ke fraud labels ka **feedback loop** taaki model seekhta rahe\n"
        "- Model + rules **saath** use karo (hybrid); model fail ho to rule fallback\n\n"
        "**D. Security & governance**\n"
        "- API par authentication, encryption (TLS), rate-limiting, audit logs\n"
        "- Customer data ki privacy (masking), access control\n"
        "- Model ke faislon ki **explainability** (SHAP/red-flags) regulator aur customer ke liye\n"
        "- Mule accounts ke liye network/graph analysis aur blacklist sharing"
    )

with tabs[4]:
    st.markdown("### Kaise pata chalega ki implemented actions kaam kar rahe hain?")
    st.markdown(
        "**1. Pehle KPIs tay karo (baseline lo):**\n"
        "- Fraud loss ₹ (mahine ka), fraud rate\n"
        "- **Recall / detection rate** (kitne fraud pakde), **Precision** (false alarm kitne)\n"
        "- **False-positive rate / customer friction** (kitne genuine customers block hue)\n"
        "- Time-to-detect aur analyst review ka load\n\n"
        "**2. Compare karne ke tareeke:**\n"
        "- **Before vs After:** action lagane se pehle aur baad ke KPIs (seasonality ka dhyan rakho)\n"
        "- **A/B test / champion-challenger:** kuch traffic par naya control, baaki par purana. Dono ke fraud loss compare karo\n"
        "- **Shadow mode:** naya model chale par block na kare, sirf log kare; baad mein asli fraud se match karo\n"
        "- **Backtesting:** purane data par naye rules/model lagakar dekho kitna fraud pakda jata\n\n"
        "**3. Statistical check:** difference real hai ya luck? (z-test / chi-square, confidence interval)\n\n"
        "**4. Continuous monitoring:** is page ke Performance log + Drift check + Retrain trigger roz/hafte chalao, dashboard aur alerts lagao.\n\n"
        "**5. Success ka matlab:** fraud loss ↓, recall stable ya ↑, false-positive ↓ ya same, customer complaints na badhein."
    )
    st.markdown("**Simple impact calculator**")
    c1, c2, c3 = st.columns(3)
    before = c1.number_input("Pehle mahine ka fraud loss (₹)", 0, 10**10, 1_000_000, 10_000)
    after = c2.number_input("Ab mahine ka fraud loss (₹)", 0, 10**10, 600_000, 10_000)
    cost = c3.number_input("Prevention ka mahine ka kharcha (₹)", 0, 10**10, 100_000, 10_000)
    saved = before - after
    a, b, c = st.columns(3)
    a.metric("Fraud loss mein kami", f"{(saved / before * 100) if before else 0:.1f}%")
    b.metric("Bachat", f"₹{saved:,.0f}")
    c.metric("Net fayda", f"₹{saved - cost:,.0f}", "ROI positive" if saved > cost else "ROI negative", delta_color="normal" if saved > cost else "inverse")

ui.next_step(12, "Agla step: Project Report (Q&A)")
