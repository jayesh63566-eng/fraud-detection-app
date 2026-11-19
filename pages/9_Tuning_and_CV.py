import plotly.express as px
import streamlit as st

from core import evaluation, modeling, state, ui

ui.setup("Tuning & Cross-Validation", "⚙️", 9, "Model ko behtar banao aur check karo ki wo bharosemand hai")
sp = state.require("split", "Model Training", "pages/7_Model_Training.py")
models = state.require("models", "Model Training", "pages/7_Model_Training.py")

ui.explain(
    "**Cross-Validation (CV):** ek hi test split par achha score luck ho sakta hai. CV data ko K hisson mein baant kar K baar "
    "train/test karta hai. Score har baar similar ho to model **stable** hai.\n\n"
    "**Hyperparameter tuning:** har model ki 'settings' hoti hain (jaise tree ki depth). Randomized search kai combinations "
    "try karke best dhundhta hai.\n\n"
    "⏱️ Bade data par ye slow ho sakta hai, isliye folds/iterations kam se shuru karo."
)

t1, t2 = st.tabs(["🔁 Cross-Validation", "🎛️ Hyperparameter tuning"])
names = [m for m in modeling.MODEL_ZOO]
cfg = state.get("train_cfg", dict(method="Class weights", ratio=0.5))

with t1:
    c1, c2 = st.columns(2)
    name = c1.selectbox("Model", names, index=names.index("Random Forest") if "Random Forest" in names else 0, key="cv_m")
    folds = c2.slider("Folds (K)", 3, 10, 5)
    st.caption(f"Imbalance method: **{cfg['method']}** (har fold ke train hisse mein hi lagta hai, leakage nahi).")
    if st.button("🔁 CV chalao", type="primary"):
        bar = st.progress(0.0, text="CV chal raha hai…")
        res = modeling.cross_validate_model(name, sp["X_train"], sp["y_train"], folds, cfg["method"], cfg["ratio"],
                                            progress=lambda p: bar.progress(p, text=f"{int(p * 100)}%"))
        state.put("cv_res", (name, res))
        bar.empty()
    cv = state.get("cv_res")
    if cv:
        nm, res = cv
        st.markdown(f"**{nm}** · {len(res)} folds")
        ui.table(res.round(4))
        mean, std = res.drop(columns="fold").mean(), res.drop(columns="fold").std()
        c = st.columns(5)
        for col, k in zip(c, ["pr_auc", "recall", "precision", "f1", "roc_auc"]):
            col.metric(k.upper(), f"{mean[k]:.3f}", f"± {std[k]:.3f}", delta_color="off")
        ui.plot(px.box(res.melt(id_vars="fold", var_name="metric", value_name="score"), x="metric", y="score", points="all"))
        if std["pr_auc"] > 0.08:
            st.warning("⚠️ Folds ke beech score bahut badalta hai. Model unstable hai ya fraud rows bahut kam hain.")
        else:
            st.success("✅ Score folds mein stable hai.")

with t2:
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Model", names, index=names.index("Random Forest") if "Random Forest" in names else 0, key="tune_m")
    n_iter = c2.slider("Kitne combinations try karein", 3, 30, 8)
    cv_k = c3.slider("CV folds", 2, 5, 3)
    scoring = st.selectbox("Optimize kis par?", ["average_precision", "recall", "f1", "roc_auc"],
                           help="average_precision = PR-AUC (fraud ke liye best)")
    with st.expander("Search space dekho"):
        st.json({k: [str(x) for x in v] for k, v in modeling.PARAM_GRIDS[name].items()})
    if st.button("🎛️ Tuning shuru karo", type="primary"):
        with st.spinner("Best settings dhundh rahe hain… (thoda time lagega)"):
            res = modeling.tune_model(name, sp, n_iter, cv_k, scoring)
        state.put("tune_res", res)
    tr = state.get("tune_res")
    if tr:
        st.success(f"Best CV score: **{tr['cv_score']:.4f}**")
        st.markdown("**Best parameters:**")
        st.json({k.replace("model__", ""): v if not hasattr(v, "item") else v.item() for k, v in tr["best_params"].items()})
        base_name = tr["name"].replace(" (tuned)", "")
        a, b = st.columns(2)
        if base_name in models:
            a.markdown("**Pehle (baseline)**")
            ui.metric_row(models[base_name]["metrics"], ("pr_auc", "recall", "precision"))
        b.markdown("**Tuned (test set par)**")
        ui.metric_row(tr["metrics"], ("pr_auc", "recall", "precision"))
        with st.expander("Saare combinations ka result"):
            ui.table(tr["cv_table"].astype({"params": str}).round(4))
        if st.button("➕ Tuned model ko leaderboard mein jodo"):
            ms = state.get("models")
            ms[tr["name"]] = tr
            state.put("models", ms)
            lb = evaluation.leaderboard(ms)
            state.put("best_model", lb.iloc[0]["model"])
            st.success(f"Jod diya. Ab best model: {lb.iloc[0]['model']}")

ui.next_step(9, "Agla step: Interpretation")
