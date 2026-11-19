import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from core import cleaning
from core import features as F
from core import plots, state, ui
from core.modeling import _cap_rows, make_split

ui.setup("Feature Selection", "🎯", 6, "Sirf kaam ke features chuno: kam features = tez, saaf, kam overfit model")
feat = state.require("feat_df", "Feature Engineering", "pages/5_Feature_Engineering.py")

ui.explain(
    "**Sab features model ko dene ki zarurat nahi.** Kuch features ek jaise hote hain (multicollinearity), kuch bekaar. "
    "Yahan 5 tareeqe hain jo batate hain kaun se features important hain: **Correlation, VIF, Random Forest importance, "
    "SelectKBest, RFE**. Last tab mein final list chuno.\n\n"
    "🔒 Ye saari calculations sirf **train hisse** par hoti hain (80/20, seed 42), taaki test data ka koi hint model ko na mile."
)

cands = F.candidate_features(feat)
base_split = make_split(feat, cands, 0.2, 42)
Xtr, ytr = _cap_rows(base_split["X_train"], base_split["y_train"], 30000, 42)
st.caption(f"Analysis {len(Xtr):,} train rows par (saare fraud rows included).")

tabs = st.tabs(["🔗 Correlation", "📏 VIF", "🌲 RF importance", "📊 SelectKBest", "🔁 RFE", "✅ Final selection"])

with tabs[0]:
    thr = st.slider("Itni se zyada correlation = 'duplicate' feature", 0.7, 0.99, 0.9, 0.01)
    ui.plot(plots.corr_fig(Xtr.assign(isFraud=ytr), cands + ["isFraud"]))
    pairs = cleaning.high_corr_pairs(Xtr, cands, thr)
    if pairs.empty:
        st.success("Koi bahut zyada correlated pair nahi.")
    else:
        st.markdown("**Bahut correlated pairs** (inme se ek hata sakte ho):")
        ui.table(pairs)

with tabs[1]:
    st.markdown("**VIF** batata hai ki ek feature baaki features se kitna predict ho jata hai. **VIF > 10** = bahut overlap.")
    vt = cleaning.vif_table(Xtr, cands)
    ui.table(vt.assign(VIF=vt["VIF"].replace(np.inf, 9999).round(1)))
    st.caption("`balanceChange`, `oldbalance`, `newbalance`, `amount` ek dusre se bane hain, isliye VIF bahut high aana normal hai. "
               "Tree models (RF, XGBoost) multicollinearity se kam pareshan hote hain, par Logistic Regression ko fark padta hai.")

with tabs[2]:
    n_est = st.slider("Trees", 50, 300, 100, 50)
    if st.button("🌲 RF importance nikalo"):
        rf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, class_weight="balanced", random_state=42).fit(Xtr, ytr)
        state.put("rf_imp", pd.DataFrame({"feature": cands, "importance": rf.feature_importances_}).sort_values("importance", ascending=False))
    imp = state.get("rf_imp")
    if imp is not None:
        ui.plot(plots.importance_fig(imp, top=20, title="Random Forest feature importance"))

with tabs[3]:
    k = st.slider("Top K", 3, len(cands), min(10, len(cands)))
    sc = st.radio("Score function", ["f_classif (ANOVA F)", "mutual_info (non-linear)"], horizontal=True)
    if st.button("📊 Score nikalo"):
        fn = f_classif if sc.startswith("f_") else mutual_info_classif
        kb = SelectKBest(fn, k=k).fit(Xtr.fillna(0), ytr)
        state.put("kbest", pd.DataFrame({"feature": cands, "score": kb.scores_, "selected": kb.get_support()}).sort_values("score", ascending=False))
    kbd = state.get("kbest")
    if kbd is not None:
        ui.table(kbd.round(3))

with tabs[4]:
    st.markdown("**RFE** ek-ek karke sabse kamzor feature hatata hai (Logistic Regression ke saath).")
    nf = st.slider("Kitne features bachane hain", 3, len(cands), min(8, len(cands)), key="rfe_n")
    if st.button("🔁 RFE chalao"):
        with st.spinner("RFE chal raha hai…"):
            Xs = StandardScaler().fit_transform(Xtr.fillna(0))
            r = RFE(LogisticRegression(max_iter=500, class_weight="balanced"), n_features_to_select=nf).fit(Xs, ytr)
        state.put("rfe", pd.DataFrame({"feature": cands, "ranking": r.ranking_, "selected": r.support_}).sort_values("ranking"))
    rfe = state.get("rfe")
    if rfe is not None:
        ui.table(rfe)

with tabs[5]:
    st.markdown("### Final features chuno")
    default = F.default_selected(cands)
    # keep the multiselect's own state valid, and let the buttons overwrite it
    st.session_state["fs_multi"] = [f for f in st.session_state.get("fs_multi", default) if f in cands] or default
    a, b = st.columns(2)
    if a.button("✨ Auto-suggest (RF importance se)"):
        imp = state.get("rf_imp")
        if imp is None:
            st.warning("Pehle 'RF importance' tab mein importance nikalo.")
        else:
            keep = imp[(imp["importance"] > 0.005) & (~imp["feature"].isin(["step", "day"]))]["feature"].tolist()
            st.session_state["fs_multi"] = keep or default
    if b.button("↩️ Default (step/day chhod kar sab)"):
        st.session_state["fs_multi"] = default
    pick = st.multiselect("Model ko ye features milenge", cands, key="fs_multi")
    st.caption(f"{len(pick)} / {len(cands)} features chune.")
    st.info("`step` aur `day` default mein hata diye: time khud fraud ki wajah nahi, aur future data mein ye kaam nahi karta.")
    if len(pick) < 2:
        st.error("Kam se kam 2 features chuno.")
    elif st.button("✅ Features confirm karo", type="primary"):
        state.reset_from("features")
        state.put("features", pick)
        st.success("Features save ho gaye.")
        st.rerun()

if state.get("features"):
    st.success(f"✅ Confirmed ({len(state.get('features'))}): " + ", ".join(state.get("features")))
    ui.next_step(6, "Agla step: Model Training")
