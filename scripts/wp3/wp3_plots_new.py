from __future__ import annotations

"""
WP3 — Plot utilities (consolidated)

This module centralizes all plotting helpers used by the notebook:
- fig2_style_svm_from_kernel: 2D projection (KernelPCA) + linear SVM visualization
- Exploratory plots comparing DRF–WL vs ITS–WL across modes and settings
- Dashboard functions that aggregate and visualize experiment results

Stable exported keys for dashboards:
  - "heatmap_kernel_mode"
  - "acc_vs_n_with_bands"
  - "diff_heatmap_drf_minus_its"
"""

from typing import Dict, Any, Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC


__all__ = [
    "fig2_style_svm_from_kernel",
    "plot_heatmaps_by_k",
    "plot_difference_heatmap",
    "plot_slope_drf_vs_its",
    "plot_drf_minus_its_bar",
    "plot_drf_vs_its_dots",
    "plot_accuracy_by_k",
    "plot_acc_curves_with_bands",
    "plot_kernel_mode_heatmap",
    "plot_drf_minus_its_difference_heatmap",
    "plot_experiment_dashboard",
]


# ------------------------------------------------------------
# Fig-2 style plotting: precomputed kernel -> 2D + linear SVM
# ------------------------------------------------------------
def fig2_style_svm_from_kernel(
    K: np.ndarray,
    y: Sequence,
    *,
    class_a=None,
    class_b=None,
    n_points_grid: int = 250,
    C: float = 1.0,
    seed: int = 42,
    title: str = "Fig-2 style SVM view (KernelPCA → 2D)",
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Pipeline:
      1) Restrict to two classes (auto-top-2 by frequency if not provided)
      2) KernelPCA (precomputed kernel) => 2D embedding
      3) Train linear SVM in 2D space
      4) Plot:
         - Colored prediction regions + decision boundary
         - Points for each class + support vectors

    Returns:
      (fig, info_dict)
    """
    y = np.asarray(y)

    # choose two classes if not provided
    uniq, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    if class_a is None or class_b is None:
        if len(uniq) < 2:
            raise ValueError("Need at least 2 classes for binary Fig-2 plot.")
        class_a = uniq[order[0]]
        class_b = uniq[order[1]]

    mask = (y == class_a) | (y == class_b)
    if mask.sum() < 3:
        raise ValueError("Too few samples for the selected classes.")

    K2 = K[np.ix_(mask, mask)]
    y2 = y[mask]
    ybin = np.where(y2 == class_a, 0, 1)

    # KernelPCA
    kpca = KernelPCA(n_components=2, kernel="precomputed", random_state=seed)
    X2d = kpca.fit_transform(K2)

    # Linear SVM in 2D
    clf = SVC(kernel="linear", C=C, random_state=seed)
    clf.fit(X2d, ybin)

    # Grid for decision contours
    x_min, x_max = X2d[:, 0].min(), X2d[:, 0].max()
    y_min, y_max = X2d[:, 1].min(), X2d[:, 1].max()
    pad_x = 0.15 * (x_max - x_min + 1e-9)
    pad_y = 0.15 * (y_max - y_min + 1e-9)

    gx = np.linspace(x_min - pad_x, x_max + pad_x, n_points_grid)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, n_points_grid)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.c_[xx.ravel(), yy.ravel()]

    pred = clf.predict(grid).reshape(xx.shape)
    dec = clf.decision_function(grid).reshape(xx.shape)

    sv_idx = clf.support_
    sv = X2d[sv_idx]

    # Compose figure
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=gx, y=gy, z=pred.astype(float),
            showscale=False,
            contours=dict(coloring="fill"),
            opacity=0.35,
            hoverinfo="skip",
            name="Predicted regions",
        )
    )
    fig.add_trace(
        go.Contour(
            x=gx, y=gy, z=dec,
            contours=dict(start=0, end=0, size=1),
            showscale=False,
            line=dict(width=3),
            hoverinfo="skip",
            name="Decision boundary (0)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X2d[ybin == 0, 0], y=X2d[ybin == 0, 1],
            mode="markers",
            name=f"class {class_a}",
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X2d[ybin == 1, 0], y=X2d[ybin == 1, 1],
            mode="markers",
            name=f"class {class_b}",
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sv[:, 0], y=sv[:, 1],
            mode="markers",
            name="support vectors",
            marker=dict(size=12, symbol="circle-open"),
        )
    )

    fig.update_layout(
        title=title + f" | classes: {class_a} vs {class_b}",
        xaxis_title="KernelPCA-1",
        yaxis_title="KernelPCA-2",
        width=900,
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    info = {"class_a": class_a, "class_b": class_b, "n_samples": int(mask.sum())}
    return fig, info


# ------------------------------------------------------------
# Heatmaps / comparisons (matplotlib + seaborn)
# ------------------------------------------------------------
def plot_heatmaps_by_k(df: pd.DataFrame, metric: str = "accuracy") -> None:
    d = df.dropna(subset=["k"]).copy()
    ks = sorted(d["k"].unique())

    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        pivot = (
            d[d["k"] == k]
            .pivot_table(index="kernel", columns="mode", values=metric, aggfunc="mean")
        )
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0.3,
            vmax=1.0,
            cbar=ax == axes[-1],
        )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Mode")
        ax.set_ylabel("Kernel")

    fig.suptitle("WP3: Mean accuracy by kernel and mode", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_difference_heatmap(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["k"]).copy()
    pivot = (
        d.pivot_table(index=["k", "mode"], columns="kernel", values="accuracy", aggfunc="mean")
        .reset_index()
    )
    if "DRF–WL" not in pivot.columns or "ITS–WL" not in pivot.columns:
        raise ValueError("Need kernel columns 'DRF–WL' and 'ITS–WL'.")
    pivot["diff"] = pivot["DRF–WL"] - pivot["ITS–WL"]
    diff_pivot = pivot.pivot(index="mode", columns="k", values="diff")

    plt.figure(figsize=(5, 4))
    sns.heatmap(diff_pivot, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Accuracy difference (DRF − ITS)")
    plt.tight_layout()
    plt.show()


def plot_slope_drf_vs_its(df: pd.DataFrame, k: Optional[int] = None, tag: Optional[str] = None) -> None:
    d = df.copy()
    if k is not None:
        d = d[d["k"] == k]
    if tag is not None:
        d = d[d["tag"] == tag]

    agg = d.groupby(["kernel", "mode"], as_index=False)["accuracy"].mean()
    p = agg.pivot(index="mode", columns="kernel", values="accuracy").reset_index()

    if "DRF–WL" not in p.columns or "ITS–WL" not in p.columns:
        raise ValueError("Need 'DRF–WL' and 'ITS–WL' in df['kernel'].")

    fig = go.Figure()
    for _, row in p.iterrows():
        fig.add_trace(
            go.Scatter(
                x=["ITS–WL", "DRF–WL"],
                y=[row["ITS–WL"], row["DRF–WL"]],
                mode="lines+markers+text",
                text=[None, row["mode"]],
                textposition="middle right",
                name=row["mode"],
                showlegend=False,
            )
        )

    title = "DRF vs ITS (mean accuracy)"
    if tag:
        title += f" | {tag}"
    if k is not None:
        title += f" | k={k}"

    fig.update_layout(title=title, yaxis_title="accuracy", xaxis_title="")
    fig.show()


def plot_drf_minus_its_bar(df: pd.DataFrame) -> None:
    d = df.copy()
    d = d[d["kernel"].isin(["DRF–WL", "ITS–WL"])]

    pivot = (
        d.groupby(["mode", "k", "n", "test_size", "kernel"])["accuracy"]
        .mean()
        .unstack("kernel")
        .reset_index()
    )
    if "DRF–WL" not in pivot.columns or "ITS–WL" not in pivot.columns:
        raise ValueError("Need 'DRF–WL' and 'ITS–WL' in df['kernel'].")

    pivot["diff"] = pivot["DRF–WL"] - pivot["ITS–WL"]
    labels = pivot.apply(lambda r: f"{r['mode']} | k={int(r['k'])}", axis=1)

    plt.figure(figsize=(7, 4))
    plt.barh(labels, pivot["diff"], color="steelblue")
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("Accuracy difference (DRF − ITS)")
    plt.title("DRF vs ITS performance difference")
    plt.tight_layout()
    plt.show()


def plot_drf_vs_its_dots(df: pd.DataFrame) -> None:
    d = df.copy()
    d = d[d["kernel"].isin(["DRF–WL", "ITS–WL"])]

    plt.figure(figsize=(7, 4))
    sns.stripplot(
        data=d,
        x="accuracy",
        y="mode",
        hue="kernel",
        dodge=True,
        size=8,
        jitter=0.15,
    )
    plt.title("DRF vs ITS accuracy distribution")
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_k(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["k"]).copy()
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=d,
        x="k",
        y="accuracy",
        hue="kernel",
        style="mode",
        markers=True,
        dashes=False,
    )
    plt.title("Accuracy vs k (shared classes)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Aggregation helpers for bands/means across runs
# ------------------------------------------------------------
def _agg_mean_std(
    df: pd.DataFrame,
    group_cols: List[str],
    metric: str = "accuracy",
) -> pd.DataFrame:
    g = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_runs"})
    )
    g[f"{metric}_std"] = g[f"{metric}_std"].fillna(0.0)
    return g


def plot_acc_curves_with_bands(
    df_results: pd.DataFrame,
    *,
    x: str = "n",
    metric: str = "accuracy",
    facet_col: str = "mode",
    color_col: str = "kernel",
    line_dash_col: Optional[str] = "k",  # k=1 vs k=2
    title: str = "Accuracy vs n (mean ± std across runs)",
) -> go.Figure:
    """
    Expects at least columns: kernel, mode, n, accuracy.
    Aggregates over seeds/runs to show mean ± std bands.
    """
    df = df_results.copy()
    needed = {x, metric, facet_col, color_col}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"df_results missing columns: {sorted(miss)}")

    group_cols = [facet_col, color_col, x]
    if line_dash_col is not None and line_dash_col in df.columns:
        group_cols.insert(2, line_dash_col)

    agg = _agg_mean_std(df, group_cols=group_cols, metric=metric)

    from plotly.subplots import make_subplots

    modes = sorted(agg[facet_col].unique().tolist())
    fig = make_subplots(
        rows=len(modes),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{facet_col}={m}" for m in modes],
        vertical_spacing=0.08,
    )

    for r, m in enumerate(modes, start=1):
        sub = agg[agg[facet_col] == m].copy()

        if line_dash_col is not None and line_dash_col in sub.columns:
            groups = sub.groupby([color_col, line_dash_col])
        else:
            groups = sub.groupby([color_col])

        for key, gdf in groups:
            gdf = gdf.sort_values(x)
            if isinstance(key, tuple):
                kernel_name, kval = key
                label = f"{kernel_name} | k={kval}"
            else:
                kernel_name = key
                label = f"{kernel_name}"

            xvals = gdf[x].to_numpy()
            ymean = gdf[f"{metric}_mean"].to_numpy()
            ystd = gdf[f"{metric}_std"].to_numpy()

            fig.add_trace(
                go.Scatter(
                    x=xvals,
                    y=ymean,
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=(r == 1),
                ),
                row=r,
                col=1,
            )

            upper = np.clip(ymean + ystd, 0.0, 1.0)
            lower = np.clip(ymean - ystd, 0.0, 1.0)

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([xvals, xvals[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    opacity=0.18,
                    line=dict(width=0),
                    hoverinfo="skip",
                    name=f"{label} band",
                    legendgroup=label,
                    showlegend=False,
                ),
                row=r,
                col=1,
            )

    fig.update_layout(
        title=title,
        height=360 * max(1, len(modes)),
        width=950,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def plot_kernel_mode_heatmap(
    df_results: pd.DataFrame,
    *,
    metric: str = "accuracy",
    title: str = "Mean accuracy by kernel × mode",
    include_k: bool = True,
) -> go.Figure:
    """
    Heatmap: y = kernel (optionally with k), x = mode, z = mean accuracy.
    """
    df = df_results.copy()
    need = {"kernel", "mode", metric}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"df_results missing columns: {sorted(miss)}")

    if include_k and "k" in df.columns:
        df["kernel_k"] = df["kernel"].astype(str) + " | k=" + df["k"].astype(str)
        row = "kernel_k"
    else:
        row = "kernel"

    agg = (
        df.groupby([row, "mode"], dropna=False)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
    )

    pivot = agg.pivot(index=row, columns="mode", values="mean").fillna(0.0)

    fig = px.imshow(
        pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        aspect="auto",
        title=title,
        zmin=0,
        zmax=1,
        text_auto=True,
    )
    fig.update_layout(width=900, height=380 + 22 * len(pivot.index))
    return fig


def plot_drf_minus_its_difference_heatmap(
    df_results: pd.DataFrame,
    *,
    metric: str = "accuracy",
    group_cols: Optional[List[str]] = None,
    title: str = "Difference heatmap: DRF–WL minus ITS–WL (mean accuracy)",
) -> go.Figure:
    """
    Builds a DRF-ITS difference heatmap across grouped settings.
    Default grouping: ["mode", "n", "test_size", "k"] if present.
    """
    df = df_results.copy()
    need = {"kernel", "mode", metric}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"df_results missing columns: {sorted(miss)}")

    if group_cols is None:
        group_cols = ["mode"]
        for c in ["n", "test_size", "k"]:
            if c in df.columns:
                group_cols.append(c)

    def _kind(k: str) -> str:
        k = str(k).lower()
        if "drf" in k:
            return "DRF"
        if "its" in k:
            return "ITS"
        return "OTHER"

    df["kernel_kind"] = df["kernel"].map(_kind)
    df = df[df["kernel_kind"].isin(["DRF", "ITS"])].copy()

    agg = (
        df.groupby(group_cols + ["kernel_kind"], dropna=False)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
    )

    piv = agg.pivot_table(index=group_cols, columns="kernel_kind", values="mean", fill_value=0.0).reset_index()
    if "DRF" not in piv.columns or "ITS" not in piv.columns:
        raise ValueError("Need both DRF and ITS runs in df_results for a difference heatmap.")

    piv["diff"] = piv["DRF"] - piv["ITS"]

    def _row_label(row) -> str:
        parts = [f"{c}={row[c]}" for c in group_cols]
        return " | ".join(parts)

    piv["setting"] = piv.apply(_row_label, axis=1)

    xcol = "n" if "n" in piv.columns else ("mode" if "mode" in piv.columns else group_cols[0])
    mat = piv.pivot_table(index="setting", columns=xcol, values="diff", fill_value=0.0)

    fig = px.imshow(
        mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        aspect="auto",
        title=title,
        text_auto=True,
    )
    fig.update_layout(width=950, height=420 + 20 * len(mat.index))
    return fig


# ------------------------------------------------------------
# Dashboard aggregator
# ------------------------------------------------------------
def plot_experiment_dashboard(
    df_results: pd.DataFrame,
    *,
    title_prefix: str = "WP3 Dashboard",
) -> Dict[str, go.Figure]:
    """
    Returns a dict with stable keys:
      - "heatmap_kernel_mode"
      - "acc_vs_n_with_bands"
      - "diff_heatmap_drf_minus_its"
    """
    figs: Dict[str, go.Figure] = {}

    figs["heatmap_kernel_mode"] = plot_kernel_mode_heatmap(
        df_results,
        title=f"{title_prefix}: mean accuracy (kernel × mode)",
        include_k=True,
    )
    figs["acc_vs_n_with_bands"] = plot_acc_curves_with_bands(
        df_results,
        title=f"{title_prefix}: accuracy vs n (mean ± std)",
    )
    figs["diff_heatmap_drf_minus_its"] = plot_drf_minus_its_difference_heatmap(
        df_results,
        title=f"{title_prefix}: DRF minus ITS (mean accuracy)",
    )
    return figs