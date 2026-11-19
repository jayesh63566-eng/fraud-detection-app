import pandas as pd
import streamlit as st

from core import evaluation, modeling, plots, state, ui

ui.setup("Model Selection & Training", "🤖", 7, "Split → imbalance handling → models train karo")
feat = state.require("feat_df", "Feature Engineering", "pages/5_Feature_Engineering.py")
features = state.require("features", "Feature Selection", "pages/6_Feature_Selection.py")

ui.explain(
    "**3 chhote steps:**\n\n"
    "1. **Split:** data ko *train* (model seekhta hai) aur *test* (model ka imtihaan) mein todo. Test ko model kabhi nahi dekhta.\n"
    "2. **Imbalance:** fraud bahut kam hota hai, to model usse ignore kar deta hai. Isko theek karne ke tareeqe chuno. "
    "Ye **sirf train par** lagta hai, test untouched rehta hai.\n"
    "3. **Train:** ek ya zyada models chuno, ek click mein sab train ho kar compare honge.",
    expanded=True,
)

t1, t2, t3 = st.tabs(["1️⃣ Split", "2️⃣ Imbalance", "3️⃣ Train models"])

with t1:
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    mode = c2.selectbox("Split type", ["Stratified random", "Time-based (step)"],
                        help="Stratified = dono hisson mein fraud ka % same. Time-based = purane data par train, naye par test (real-life jaisa).")
    seed = c3.number_input("Random seed", 0, 9999, 42)
    if mode.startswith("Time") and "step" not in feat.columns:
        st.error("Time-based ke liye `step` column chahiye.")
        mode = "Stratified random"
    if st.button("✂️ Split banao", type="primary"):
        state.reset_from("split")
        state.put("split", modeling.make_split(feat, features, test_size, int(seed), mode))
        st.rerun()
    sp = state.get("split")
    if sp is not None:
        a, b = st.columns(2)
        a.metric("Train rows", f"{len(sp['X_train']):,}", f"fraud {int(sp['y_train'].sum()):,} ({sp['y_train'].mean() * 100:.2f}%)")
        b.metric("Test rows", f"{len(sp['X_test']):,}", f"fraud {int(sp['y_test'].sum()):,} ({sp['y_test'].mean() * 100:.2f}%)")
        st.success(f"✅ Split ready ({sp['mode']}, {len(sp['features'])} features)")
        if sp["y_test"].sum() < 20:
            st.warning("⚠️ Test mein 20 se kam fraud hain. Metrics unreliable honge. Zyada data ya bada test size lo.")

with t2:
    st.markdown("**Imbalance handling ka tareeka chuno:**")
    method = st.radio("Method", list(modeling.RESAMPLE_METHODS), index=1, format_func=lambda m: f"{m}: {modeling.RESAMPLE_METHODS[m]}")
    ratio = 0.5
    if method in ("SMOTE", "Random oversample", "Random undersample"):
        ratio = st.slider("Target ratio (fraud : normal)", 0.05, 1.0, 0.5, 0.05,
                          help="1.0 = barabar. 0.5 = har 2 normal par 1 fraud.")
    state.put("train_cfg", dict(method=method, ratio=ratio))
    ui.tip("Class weights sabse safe hai. SMOTE tab accha jab fraud rows kaafi hon. Undersample se data bahut kam ho jata hai.")
    sp = state.get("split")
    if sp is not None and method != "None":
        with st.expander("👀 Resampling ke baad train data kaisa dikhega"):
            Xr, yr = modeling.resample(sp["X_train"], sp["y_train"], method, ratio)
            a, b = st.columns(2)
            a.metric("Pehle", f"{len(sp['y_train']):,} rows", f"fraud {sp['y_train'].mean() * 100:.2f}%")
            b.metric("Baad mein", f"{len(yr):,} rows", f"fraud {yr.mean() * 100:.2f}%")

with t3:
    sp = state.get("split")
    if sp is None:
        st.warning("Pehle '1️⃣ Split' tab mein split banao.")
        st.stop()
    cfg = state.get("train_cfg", dict(method="Class weights", ratio=0.5))
    st.markdown(f"Imbalance method: **{cfg['method']}**")
    st.markdown("**Models chuno** (jitne zyada, utna time):")
    chosen = []
    cols = st.columns(2)
    for i, (name, z) in enumerate(modeling.MODEL_ZOO.items()):
        with cols[i % 2]:
            if st.checkbox(name, value=name in modeling.DEFAULT_MODELS, help=z["desc"], key=f"m_{name}"):
                chosen.append(name)
            st.caption(z["desc"])
    if "XGBoost" not in modeling.MODEL_ZOO:
        st.caption("XGBoost install nahi hai (`pip install xgboost`).")

    if st.button("🚀 Train karo", type="primary", disabled=not chosen):
        res = {}
        bar = st.progress(0.0, text="Shuru…")
        for i, name in enumerate(chosen):
            bar.progress(i / len(chosen), text=f"{name} train ho raha hai…")
            try:
                res[name] = modeling.train_model(name, sp, cfg["method"], cfg["ratio"])
            except Exception as e:
                st.error(f"{name} fail: {e}")
        bar.progress(1.0, text="Ho gaya ✅")
        if res:
            old = state.get("models") or {}
            old.update(res)
            state.reset_from("models")
            state.put("models", old)
            lb = evaluation.leaderboard(old)
            state.put("best_model", lb.iloc[0]["model"])
        st.rerun()

    models = state.get("models")
    if models:
        lb = evaluation.leaderboard(models)
        st.subheader("🏆 Leaderboard (PR-AUC se sorted)")
        ui.table(lb.round(4))
        ui.plot(plots.leaderboard_fig(lb))
        st.success(f"Abhi tak ka best: **{lb.iloc[0]['model']}**. Detail Evaluation page par dekho.")
        ui.next_step(7, "Agla step: Evaluation")
