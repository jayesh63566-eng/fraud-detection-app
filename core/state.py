"""Shared session-state helpers so every page reads/writes the same pipeline objects."""
import streamlit as st

# Pipeline objects, in order. Each page depends on the ones before it.
KEYS = ["raw_df", "clean_df", "feat_df", "features", "split", "models", "best_model"]
LABELS = {
    "raw_df": "Data loaded",
    "clean_df": "Preprocessed",
    "feat_df": "Features built",
    "features": "Features selected",
    "split": "Train/test split",
    "models": "Models trained",
    "best_model": "Best model chosen",
}


def get(key, default=None):
    return st.session_state.get(key, default)


def put(key, value):
    st.session_state[key] = value


def reset_from(key):
    """Clear `key` and everything after it (when an earlier step is redone)."""
    i = KEYS.index(key)
    for k in KEYS[i:]:
        st.session_state.pop(k, None)
    st.session_state.pop("feat_opts", None) if key in ("raw_df", "clean_df", "feat_df") else None


def require(key, hint, page=None):
    """Stop the page with a friendly warning if an earlier step is not done."""
    if get(key) is None:
        st.warning(f"⚠️ Pehle ye step complete karo: **{hint}**")
        if page:
            st.page_link(page, label=f"➡️ {hint} par jao", icon="👉")
        st.stop()
    return get(key)


def progress():
    return {k: get(k) is not None for k in KEYS}
