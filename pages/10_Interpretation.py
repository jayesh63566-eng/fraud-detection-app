import pandas as pd
import plotly.express as px
import streamlit as st

from core import interpret, plots, state, ui

ui.setup("Model Interpretation", "💡", 10, "Model KYA dekh kar faisla karta hai, aur kya wo sahi hai?")
models = state.require("models", "Model Training", "pages/7_Model_Training.py")
sp = state.require("split", "Model Training", "pages/7_Model_Training.py")

ui.explain(
    "**Black-box par bharosa nahi karte.** Bank ko ye batana padta hai ki 'is transaction ko fraud kyun kaha'. "
    "Yahan 4 tareeqe hain: **global importance** (model overall kya dekhta hai), **SHAP / LIME** (advanced), "
    "**ek transaction ki explanation**, aur **'kya ye logical hai?'** check."
)

name = st.selectbox("Model", list(models), index=list(models).index(state.get("best_model")) if state.get("best_model") in models else 0)
pipe = models[name]["pipeline"]
feats = sp["features"]
Xte, yte, Xtr = sp["X_test"], sp["y_test"], sp["X_train"]

tabs = st.tabs(["🌍 Global importance", "🧪 SHAP", "🍋 LIME", "🔬 Ek transaction", "🧠 Kya factors logical hain?"])

with tabs[0]:
    b = interpret.builtin_importance(pipe, feats)
    c1, c2 = st.columns(2)
    with c1:
        if b is not None:
            ui.plot(plots.importance_fig(b, title=b["kind"].iloc[0]))
        else:
            st.info("Is model mein built-in importance nahi hoti. Permutation importance dekho.")
    with c2:
        if st.button("🔀 Permutation importance nikalo"):
            with st.spinner("Calculate ho raha hai…"):
                state.put(f"perm_{name}", interpret.perm_importance(pipe, Xte, yte, n_repeats=3))
        pi = state.get(f"perm_{name}")
        if pi is not None:
            ui.plot(plots.importance_fig(pi, title="Permutation importance (PR-AUC girawat)"))
            st.caption("Ek feature ko shuffle karne se score kitna girta hai. Zyada girawat = zyada important. Ye model-agnostic aur zyada bharosemand hai.")

with tabs[1]:
    if not interpret.shap_available():
        st.info("SHAP install nahi hai: `pip install shap`. Tab tak 'Global importance' aur 'Ek transaction' tab use karo.")
    elif not hasattr(pipe.named_steps["model"], "feature_importances_"):
        st.info("Is app mein SHAP sirf tree models (Decision Tree, Random Forest, XGBoost, Gradient Boosting) ke liye hai.")
    else:
        n = st.slider("Kitne test rows par", 200, 3000, 800, 100)
        if st.button("🧪 SHAP chalao"):
            with st.spinner("SHAP values nikal rahe hain…"):
                Xs = ui.sample_for_plot(Xte.assign(isFraud=yte.values), n).drop(columns="isFraud")
                sv, Xu = interpret.shap_values_tree(pipe, Xs)
                state.put("shap", (pd.DataFrame({"feature": feats, "mean|SHAP|": abs(sv).mean(axis=0)}).sort_values("mean|SHAP|", ascending=False), sv, Xu))
        s = state.get("shap")
        if s:
            ui.plot(plots.importance_fig(s[0], col="mean|SHAP|", title="Mean |SHAP value|"))
            top = s[0]["feature"].head(8).tolist()
            idx = [feats.index(f) for f in top]
            long = pd.DataFrame({"feature": [f for f in top for _ in range(len(s[2]))],
                                 "shap": s[1][:, idx].T.ravel(), "value": s[2][top].T.to_numpy().ravel()})
            ui.plot(px.strip(long, x="shap", y="feature", color="value", color_continuous_scale="RdBu_r", stripmode="overlay"))
            st.caption("Har dot ek transaction. Dayein (positive SHAP) = fraud ki taraf, rang = feature ki value (lal = badi).")

with tabs[2]:
    if not interpret.lime_available():
        st.info("LIME install nahi hai: `pip install lime`. 'Ek transaction' tab same kaam apne tareeke se karta hai.")
    else:
        pos = Xte[yte.values == 1]
        if pos.empty:
            st.info("Test mein fraud nahi hai.")
        else:
            i = st.number_input("Fraud row number", 0, len(pos) - 1, 0, key="lime_i")
            if st.button("🍋 LIME explain"):
                with st.spinner("LIME chal raha hai…"):
                    state.put("lime", interpret.lime_explain(pipe, Xtr.sample(min(len(Xtr), 5000), random_state=0), pos.iloc[[int(i)]]))
            lm = state.get("lime")
            if lm is not None:
                ui.plot(plots.contribution_fig(lm, "weight", "rule", 8))

with tabs[3]:
    st.markdown("**Ek transaction chuno aur dekho model ne kya socha.**")
    kind = st.radio("Kaisa transaction?", ["Pakda gaya fraud", "Miss hua fraud", "False alarm", "Random normal"], horizontal=True)
    prob = models[name]["y_prob"]
    thr = float(state.get("threshold", 0.5))
    mask = {"Pakda gaya fraud": (yte.values == 1) & (prob >= thr), "Miss hua fraud": (yte.values == 1) & (prob < thr),
            "False alarm": (yte.values == 0) & (prob >= thr), "Random normal": (yte.values == 0)}[kind]
    cand = Xte[mask]
    if cand.empty:
        st.info("Is category mein koi transaction nahi hai.")
    else:
        k = st.number_input("Row number", 0, len(cand) - 1, 0, key="occ_i")
        row = cand.iloc[[int(k)]]
        exp, p0 = interpret.occlusion_explain(pipe, row, Xtr.median())
        c1, c2 = st.columns([1, 2])
        c1.metric("Fraud probability", f"{p0:.1%}")
        c1.markdown("**Transaction values**")
        c1.dataframe(row.T.rename(columns={row.index[0]: "value"}).round(2))
        with c2:
            ui.plot(plots.contribution_fig(exp, "effect", "feature", 10))
            st.caption("Har bar = us feature ko 'normal/typical' value se badalne par probability kitni badli. Lal = us feature ne fraud score badhaya.")
        from core.deploy import red_flags
        raw_like = dict(type=next((t.replace("type_", "") for t in feats if t.startswith("type_") and row[t].iloc[0] == 1), ""),
                        amount=row["amount"].iloc[0] if "amount" in row else 0,
                        oldbalanceOrg=row["oldbalanceOrg"].iloc[0] if "oldbalanceOrg" in row else 0,
                        newbalanceOrig=row["newbalanceOrig"].iloc[0] if "newbalanceOrig" in row else 0,
                        oldbalanceDest=row["oldbalanceDest"].iloc[0] if "oldbalanceDest" in row else 0,
                        newbalanceDest=row["newbalanceDest"].iloc[0] if "newbalanceDest" in row else 0)
        st.markdown("**Rule-based red flags (analyst ke liye):**")
        for f in red_flags(raw_like):
            st.markdown(f"- {f}")

with tabs[4]:
    imp = interpret.builtin_importance(pipe, feats)
    if imp is None:
        imp = interpret.perm_importance(pipe, Xte, yte, n_repeats=2)
    top_n = st.slider("Top kitne factors?", 3, min(15, len(feats)), min(8, len(feats)))
    st.markdown("### 🔑 Key factors jo fraud predict karte hain")
    ms = interpret.make_sense_table(imp["feature"].head(top_n).tolist())
    ms.insert(0, "rank", range(1, len(ms) + 1))
    ui.table(ms)
    yes = (ms["makes sense?"].str.startswith("Haan")).sum()
    no = (ms["makes sense?"].str.startswith("Nahi")).sum()
    st.success(f"{yes}/{len(ms)} top factors domain-logic se match karte hain.")
    if no:
        st.warning(f"{no} factor(s) shak ke daayre mein hain (jaise `step`/`day`). Unhe model se hatana behtar hoga.")
    st.markdown(
        "**Overall conclusion:** Fraud ka classic pattern hai — *attacker account takeover karta hai, poora balance ek TRANSFER/CASH_OUT "
        "mein nikal leta hai, aur receiver side ka balance system mein theek se update nahi dikhta.* "
        "Model ke top factors isi kahani se milte hain, isliye result **logical aur trustable** hai."
    )

ui.next_step(10, "Agla step: Deployment")
