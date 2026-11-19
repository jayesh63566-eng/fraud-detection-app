import streamlit as st

from core import state, ui
from core.loaders import KAGGLE_SLUG, make_demo_data

ui.setup("Fraud Detection System", "🏠", 0, "MSc Data Science · Marwadi University · 2024–25 · Jayesh Parmar & Aaditya Patadiya")

st.markdown(
    "Ye app **poora fraud detection workflow** step-by-step sikhata aur chalata hai: "
    "data load karna → saaf karna → charts → features → model train → evaluate → deploy → monitor. "
    "Har page par ek chhoti **'ℹ️ Ye kya hai?'** guide hai, aur neeche **'Agla step'** button."
)

# ---------------------------------------------------------------- quick start
st.subheader("🚀 Quick start")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**1️⃣ Data lao**")
    st.caption("Apni CSV, Google Drive link, Kaggle API, ya demo data.")
    st.page_link("pages/1_Data_Collection.py", label="Data Collection kholo", icon="📥")
with c2:
    st.markdown("**2️⃣ Seedha try karna hai?**")
    st.caption("20,000 rows ka synthetic PaySim-jaisa data.")
    if st.button("⚡ Demo data load karo"):
        state.reset_from("raw_df")
        state.put("raw_df", make_demo_data())
        state.put("data_source", "Demo (synthetic)")
        st.success("Demo data load ho gaya. Ab left menu se 'EDA' kholo.")
        st.rerun()
with c3:
    st.markdown("**3️⃣ Result dekho**")
    st.caption("Saare 8 project sawaalon ke jawab automatically.")
    st.page_link("pages/13_Project_Report.py", label="Project Report kholo", icon="📝")

prog = state.progress()
st.progress(sum(prog.values()) / len(prog), text=f"Pipeline {sum(prog.values())}/{len(prog)} steps complete")

st.divider()
tabs = st.tabs(["🎯 Objective", "🔄 Workflow", "👥 Users & Tools", "🗂️ Dataset", "📚 Libraries", "❓ 8 Project Questions"])

with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.subheader("Objective")
        st.markdown(
            "- Fraudulent transactions ko **real-time** detect karna\n"
            "- Machine learning se **false positives kam** karke accuracy badhana\n"
            "- Security badhana aur **financial loss kam** karna"
        )
    with b:
        st.subheader("Problem definition")
        st.markdown(
            "- Fraud se banking, e-commerce aur insurance mein bada nuksaan hota hai\n"
            "- Rule-based systems naye fraud patterns ke saath adapt nahi karte\n"
            "- Automated, scalable, data-driven solution ki zarurat hai"
        )
    st.subheader("Problem type")
    st.markdown("**Binary classification** (0 = normal, 1 = fraud) · **Highly imbalanced** (fraud ~0.1%–2%) · "
                "Isliye accuracy nahi, **PR-AUC, Recall, Precision** dekhte hain.")

with tabs[1]:
    st.subheader("13-step ML workflow  →  is app ke pages")
    st.graphviz_chart("""
    digraph G {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EAF2F8", fontname="Helvetica"];
      A [label="1 Data\\nCollection"]; B [label="2 EDA &\\nVisualization"]; C [label="3 Preprocessing"];
      D [label="4 Feature\\nEngineering"]; E [label="5 Feature\\nSelection"]; F [label="6 Model\\nTraining"];
      G [label="7 Evaluation"]; H [label="8 Tuning &\\nCross-Val"]; I [label="9 Interpretation"];
      J [label="10 Deployment"]; K [label="11 Monitoring"];
      A->B->C->D->E->F->G->H->I->J->K;
    }
    """)
    st.markdown(
        "Report ke 13 components: Problem Definition · Data Collection · EDA · Preprocessing · Feature Selection · "
        "Model Selection · Training · Evaluation · Hyperparameter Tuning · Cross-Validation · Interpretation · "
        "Deployment · Monitoring."
    )

with tabs[2]:
    a, b = st.columns(2)
    with a:
        st.subheader("Targeted users")
        st.markdown(
            "- **Banks:** fraud analysts, risk managers\n"
            "- **E-commerce:** payment processors\n"
            "- **End-users:** transparent fraud notifications"
        )
    with b:
        st.subheader("Tools")
        st.markdown(
            "**Python:** pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, SHAP/LIME\n\n"
            "**Visualization:** Plotly, matplotlib, seaborn\n\n"
            "**App & deploy:** Streamlit, FastAPI, Docker"
        )

with tabs[3]:
    st.subheader("Dataset design")
    st.markdown(f"Kaggle: `{KAGGLE_SLUG}` · Report ke hisaab se **64,442 rows × 11 columns**")
    st.table({
        "Column": ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
                   "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"],
        "Matlab": ["Time unit (1 step = 1 ghanta)", "CASH_IN / CASH_OUT / DEBIT / PAYMENT / TRANSFER",
                   "Transaction amount", "Sender ID", "Sender ka balance pehle", "Sender ka balance baad mein",
                   "Receiver ID", "Receiver ka balance pehle", "Receiver ka balance baad mein",
                   "TARGET: 1 = fraud", "Purane system ne flag kiya (bade TRANSFER)"],
    })

with tabs[4]:
    st.subheader("Libraries (report se)")
    st.table({
        "Library": ["NumPy", "Pandas", "Matplotlib / Seaborn / Plotly", "scikit-learn", "imbalanced-learn (SMOTE)",
                    "statsmodels (VIF)", "AutoGluon", "SHAP / LIME", "FastAPI", "Streamlit"],
        "Kaam": ["Numerical arrays", "Data cleaning & manipulation", "Charts", "Models, metrics, scaling, CV",
                 "Imbalance handling", "Multicollinearity check", "AutoML (optional, notebook mein)",
                 "Model explanation", "Prediction API", "Ye web app"],
    })
    st.caption("Is app mein VIF aur SMOTE ke apne implementations bhi hain, to imblearn/statsmodels na ho tab bhi chalta hai.")

with tabs[5]:
    st.subheader("Project ke 8 sawaal  →  jawab kahan milenge")
    st.markdown(
        "1. **Data cleaning** (missing, outliers, multicollinearity) → pages *Preprocessing* + *Feature Selection*\n"
        "2. **Model ka elaboration** → *Model Training* + *Project Report*\n"
        "3. **Variables kaise chune** → *Feature Selection*\n"
        "4. **Performance demonstrate karna** → *Evaluation* + *Tuning & CV*\n"
        "5. **Key fraud factors** → *Interpretation*\n"
        "6. **Kya factors logical hain?** → *Interpretation → Do factors make sense?*\n"
        "7. **Infrastructure update mein prevention** → *Monitoring & Prevention*\n"
        "8. **Kaise pata chalega ki actions kaam kar rahe hain** → *Monitoring & Prevention*"
    )

ui.next_step(0, "Shuru karo: Data Collection")
