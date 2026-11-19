"""Plotly figure builders used by several pages."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

FRAUD_COLORS = {0: "#2E86AB", 1: "#E4572E", "0": "#2E86AB", "1": "#E4572E"}


def confusion_fig(m: dict):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    txt = [[f"Sahi normal<br><b>{m['tn']:,}</b>", f"False alarm<br><b>{m['fp']:,}</b>"],
           [f"Fraud miss<br><b>{m['fn']:,}</b>", f"Fraud pakda<br><b>{m['tp']:,}</b>"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted fraud"], y=["Actual normal", "Actual fraud"],
                               text=txt, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(autorange="reversed"))
    return fig


def roc_fig(models: dict, y_true):
    fig = go.Figure()
    for name, r in models.items():
        fpr, tpr, _ = roc_curve(y_true, r["y_prob"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {r['metrics']['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Random"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate (recall)",
                      height=420, margin=dict(l=10, r=10, t=30, b=10), title="ROC curve")
    return fig


def pr_fig(models: dict, y_true):
    fig = go.Figure()
    for name, r in models.items():
        p, rc, _ = precision_recall_curve(y_true, r["y_prob"])
        fig.add_trace(go.Scatter(x=rc, y=p, mode="lines", name=f"{name} (AP {r['metrics']['pr_auc']:.3f})"))
    base = float(np.mean(y_true))
    fig.add_hline(y=base, line_dash="dash", line_color="grey", annotation_text="Random guess")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=420,
                      margin=dict(l=10, r=10, t=30, b=10), title="Precision-Recall curve")
    return fig


def threshold_fig(tbl: pd.DataFrame, current: float):
    fig = go.Figure()
    for c in ("precision", "recall", "f1"):
        fig.add_trace(go.Scatter(x=tbl["threshold"], y=tbl[c], mode="lines", name=c))
    fig.add_vline(x=current, line_dash="dash", line_color="black", annotation_text="current")
    fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=360, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def cost_fig(tbl: pd.DataFrame, current: float):
    fig = px.line(tbl, x="threshold", y="cost")
    fig.add_vline(x=current, line_dash="dash", line_color="black")
    best = tbl.loc[tbl["cost"].idxmin()]
    fig.add_trace(go.Scatter(x=[best["threshold"]], y=[best["cost"]], mode="markers",
                             marker=dict(size=12, color="#E4572E"), name="Minimum cost"))
    fig.update_layout(height=320, yaxis_title="Total business cost", margin=dict(l=10, r=10, t=30, b=10))
    return fig


def leaderboard_fig(lb: pd.DataFrame):
    m = lb.melt(id_vars="model", value_vars=["pr_auc", "recall", "precision", "f1"], var_name="metric")
    fig = px.bar(m, x="model", y="value", color="metric", barmode="group")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), yaxis_range=[0, 1.05])
    return fig


def importance_fig(df: pd.DataFrame, col="importance", top=15, title=""):
    d = df.head(top).iloc[::-1]
    fig = px.bar(d, x=col, y="feature", orientation="h", title=title)
    fig.update_layout(height=max(300, 26 * len(d) + 80), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def corr_fig(df: pd.DataFrame, cols):
    c = df[cols].corr().round(2)
    fig = px.imshow(c, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=max(420, 34 * len(cols)), margin=dict(l=10, r=10, t=30, b=10))
    return fig


def contribution_fig(df: pd.DataFrame, col: str, label_col="feature", top=10):
    d = df.head(top).iloc[::-1].copy()
    d["direction"] = np.where(d[col] >= 0, "fraud ki taraf", "normal ki taraf")
    fig = px.bar(d, x=col, y=label_col, color="direction", orientation="h",
                 color_discrete_map={"fraud ki taraf": "#E4572E", "normal ki taraf": "#2E86AB"})
    fig.update_layout(height=max(300, 30 * len(d) + 80), margin=dict(l=10, r=10, t=30, b=10))
    return fig
