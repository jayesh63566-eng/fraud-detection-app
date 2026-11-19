"""Common page chrome and small helpers that make every page look and behave the same."""
import pandas as pd
import streamlit as st

from core import state

PAGES = [
    ("app.py", "Overview", "🏠"),
    ("pages/1_Data_Collection.py", "Data Collection", "📥"),
    ("pages/2_EDA.py", "EDA", "🔍"),
    ("pages/3_Visualization.py", "Visualization", "📊"),
    ("pages/4_Preprocessing.py", "Preprocessing", "🧹"),
    ("pages/5_Feature_Engineering.py", "Feature Engineering", "🛠️"),
    ("pages/6_Feature_Selection.py", "Feature Selection", "🎯"),
    ("pages/7_Model_Training.py", "Model Training", "🤖"),
    ("pages/8_Evaluation.py", "Evaluation", "📈"),
    ("pages/9_Tuning_and_CV.py", "Tuning & Cross-Validation", "⚙️"),
    ("pages/10_Interpretation.py", "Interpretation", "💡"),
    ("pages/11_Deployment.py", "Deployment", "🚀"),
    ("pages/12_Monitoring.py", "Monitoring & Prevention", "🛡️"),
    ("pages/13_Project_Report.py", "Project Report (Q&A)", "📝"),
]


def setup(title: str, icon: str, index: int | None = None, goal: str = ""):
    """Page config + sidebar tracker + header. `index` = position in PAGES (for the progress bar)."""
    st.set_page_config(page_title=f"{title} · Fraud Detection", page_icon=icon, layout="wide")
    with st.sidebar:
        st.markdown("### 🧭 Pipeline progress")
        prog = state.progress()
        done = sum(prog.values())
        st.progress(done / len(prog))
        for k, ok in prog.items():
            st.write(("✅ " if ok else "⬜ ") + state.LABELS[k])
        st.divider()
        if state.get("raw_df") is None:
            if st.button("⚡ Demo data se shuru karo", help="Synthetic PaySim-jaisa data, 20,000 rows"):
                from core.loaders import make_demo_data
                state.reset_from("raw_df")
                state.put("raw_df", make_demo_data())
                state.put("data_source", "Demo (synthetic)")
                st.rerun()
        else:
            st.caption(f"📦 Data: {state.get('data_source', 'loaded')} · {len(state.get('raw_df')):,} rows")
    st.title(f"{icon} {title}")
    if goal:
        st.caption(goal)
    if index is not None:
        st.progress((index) / (len(PAGES) - 1), text=f"Step {index} of {len(PAGES) - 1}")


def explain(text: str, title: str = "ℹ️ Ye kya hai? (simple bhasha mein)", expanded: bool = False):
    with st.expander(title, expanded=expanded):
        st.markdown(text)


def tip(text: str):
    st.info("💡 " + text)


def next_step(index: int, label: str | None = None):
    """Big 'next page' link at the bottom of every page."""
    if index + 1 < len(PAGES):
        path, name, icon = PAGES[index + 1]
        st.divider()
        st.page_link(path, label=label or f"Agla step: {name}", icon="➡️")


def plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:  # older Streamlit: width must be int
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df, height=None, key=None):
    kw = {}
    if isinstance(df, pd.DataFrame):
        kw["hide_index"] = True
    if height is not None:      # newer Streamlit rejects height=None
        kw["height"] = height
    if key is not None:
        kw["key"] = key
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:  # older Streamlit: width must be int
        st.dataframe(df, use_container_width=True, **kw)


def download_df(df: pd.DataFrame, filename: str, label="⬇️ CSV download karo"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


def sample_for_plot(df: pd.DataFrame, n: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Keep every fraud row, sample the rest, so charts stay fast but frauds stay visible."""
    if len(df) <= n:
        return df
    if "isFraud" in df.columns:
        pos = df[df["isFraud"] == 1]
        neg = df[df["isFraud"] == 0].sample(max(n - len(pos), 1), random_state=seed)
        return pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    return df.sample(n, random_state=seed)


def working_df():
    """Most processed dataframe available (feat > clean > raw)."""
    for k in ("feat_df", "clean_df", "raw_df"):
        if state.get(k) is not None:
            return state.get(k), k
    return None, None


def metric_row(metrics: dict, keys=("pr_auc", "recall", "precision", "f1", "roc_auc")):
    cols = st.columns(len(keys))
    for c, k in zip(cols, keys):
        v = metrics.get(k)
        c.metric(k.replace("_", " ").upper(), "—" if v is None or v != v else f"{v:.3f}")