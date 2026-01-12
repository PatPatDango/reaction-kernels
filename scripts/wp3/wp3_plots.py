import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from typing import Dict, Any, Optional, Iterable
import matplotlib.pyplot as plt
import seaborn as sns


def plot_experiment_results_old(df: pd.DataFrame):
    """
    Create a compact set of plots for WP3 experiments.

    Required columns:
      - kernel: str (e.g., 'DRF–WL', 'ITS–WL')
      - mode: str (e.g., 'edge', 'vertex', 'sp')
      - n: int
      - test_size: float
      - accuracy: float

    Optional:
      - runtime_sec: float
    Returns: dict of plotly figures
    """
    required = {"kernel", "mode", "n", "test_size", "accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"plot_experiment_results: missing columns {missing}. Have: {list(df.columns)}")

    # Make sure types are sensible
    d = df.copy()
    d["n"] = d["n"].astype(int)
    d["test_size"] = d["test_size"].astype(float)
    d["accuracy"] = d["accuracy"].astype(float)

    figs = {}

    # ---- Plot A: Baseline (best row per kernel for a chosen baseline filter) ----
    # We simply show overall best accuracy per kernel in the provided df
    base = (
        d.groupby("kernel", as_index=False)["accuracy"]
        .max()
        .sort_values("accuracy", ascending=False)
    )
    figs["baseline_best_per_kernel"] = px.bar(
        base, x="kernel", y="accuracy",
        title="Best Accuracy per Kernel (across provided experiments)",
        text="accuracy",
    )

    # ---- Plot B: Mode comparison (grouped bar) ----
    # If multiple n/test_size exist, we aggregate with mean to keep it readable
    mode_cmp = (
        d.groupby(["kernel", "mode"], as_index=False)["accuracy"]
        .mean()
        .sort_values(["kernel", "mode"])
    )
    figs["mode_comparison"] = px.bar(
        mode_cmp, x="mode", y="accuracy", color="kernel", barmode="group",
        title="Accuracy by Feature Mode (mean over runs)",
    )

    # ---- Plot C: Dataset size effect (line plot) ----
    size_cmp = (
        d.groupby(["kernel", "mode", "n"], as_index=False)["accuracy"]
        .mean()
        .sort_values("n")
    )
    figs["dataset_size_effect"] = px.line(
        size_cmp, x="n", y="accuracy", color="kernel", line_dash="mode",
        markers=True,
        title="Accuracy vs Dataset Size (mean over runs)",
    )

    # ---- Plot D: Train/Test split effect (line plot) ----
    split_cmp = (
        d.groupby(["kernel", "mode", "test_size"], as_index=False)["accuracy"]
        .mean()
        .sort_values("test_size")
    )
    figs["split_effect"] = px.line(
        split_cmp, x="test_size", y="accuracy", color="kernel", line_dash="mode",
        markers=True,
        title="Accuracy vs Test Size (mean over runs)",
    )

    # ---- Plot E (optional): Runtime ----
    if "runtime_sec" in d.columns:
        rt = (
            d.groupby(["kernel", "mode"], as_index=False)["runtime_sec"]
            .mean()
            .sort_values("runtime_sec", ascending=False)
        )
        figs["runtime_by_mode"] = px.bar(
            rt, x="mode", y="runtime_sec", color="kernel", barmode="group",
            title="Runtime by Kernel/Mode (mean over runs)",
        )

    return figs

def plot_experiment_results(df: pd.DataFrame, title: str = "WP3 SVM Experiments Dashboard"):
    """
    Expects df with columns at least:
    ['tag','kernel','mode','n','test_size','accuracy','subset_ids']
    """

    d = df.copy()

    # --- make nicer labels / ensure types ---
    if "mode" in d.columns:
        d["mode"] = d["mode"].astype(str)
    if "kernel" in d.columns:
        d["kernel"] = d["kernel"].astype(str)
    if "tag" in d.columns:
        d["tag"] = d["tag"].astype(str)

    # Convert subset_ids to readable string (optional, for hover)
    if "subset_ids" in d.columns:
        d["subset_ids_str"] = d["subset_ids"].apply(lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x))

    # -----------------------------
    # Plot A: Heatmap (Kernel x Mode)
    # -----------------------------
    # Aggregate across tags/n/test_size if mixed: mean accuracy
    heat = (
        d.groupby(["kernel", "mode"], as_index=False)["accuracy"]
        .mean()
        .pivot(index="kernel", columns="mode", values="accuracy")
    )
    fig_heat = px.imshow(
        heat,
        text_auto=".3f",
        aspect="auto",
        title="Mean Accuracy Heatmap (Kernel × Mode)",
    )
    fig_heat.update_layout(margin=dict(l=20, r=20, t=60, b=20))

    # -----------------------------
    # Plot B: Accuracy vs n (facet by mode)
    # -----------------------------
    if "n" in d.columns and d["n"].notna().any():
        d_n = d.dropna(subset=["n"]).copy()
        d_n["n"] = d_n["n"].astype(int)

        fig_n = px.line(
            d_n.sort_values("n"),
            x="n",
            y="accuracy",
            color="kernel",
            markers=True,
            facet_col="mode",
            facet_col_wrap=3,
            hover_data=["tag", "test_size", "subset_ids_str"],
            title="Accuracy vs Dataset Size (n) — faceted by mode",
        )
        fig_n.update_yaxes(range=[0, 1])
        fig_n.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    else:
        fig_n = go.Figure()
        fig_n.update_layout(title="Accuracy vs Dataset Size (n) — not available (missing column 'n')")

    # -----------------------------
    # Plot C: Accuracy vs test_size (facet by mode)
    # -----------------------------
    if "test_size" in d.columns and d["test_size"].notna().any():
        d_ts = d.dropna(subset=["test_size"]).copy()
        d_ts["test_size"] = d_ts["test_size"].astype(float)

        fig_ts = px.line(
            d_ts.sort_values("test_size"),
            x="test_size",
            y="accuracy",
            color="kernel",
            markers=True,
            facet_col="mode",
            facet_col_wrap=3,
            hover_data=["tag", "n", "subset_ids_str"],
            title="Accuracy vs Test Split (test_size) — faceted by mode",
        )
        fig_ts.update_yaxes(range=[0, 1])
        fig_ts.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    else:
        fig_ts = go.Figure()
        fig_ts.update_layout(title="Accuracy vs Test Split — not available (missing column 'test_size')")

    # -----------------------------
    # Plot D: Ranking Table (Top Runs)
    # -----------------------------
    top = d.sort_values("accuracy", ascending=False).head(15).copy()
    cols = ["tag", "kernel", "mode", "n", "test_size", "accuracy", "subset_ids_str"]
    cols = [c for c in cols if c in top.columns]

    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(values=cols, align="left"),
                cells=dict(values=[top[c].tolist() for c in cols], align="left"),
            )
        ]
    )
    fig_table.update_layout(title="Top 15 Runs (highest accuracy)")

    # -----------------------------
    # Plot E (optional but cool): Scatter "Pareto-ish"
    # accuracy vs n, colored by kernel, symbol by mode
    # -----------------------------
    if "n" in d.columns and d["n"].notna().any():
        fig_scatter = px.scatter(
            d_n,
            x="n",
            y="accuracy",
            color="kernel",
            symbol="mode",
            hover_data=["tag", "test_size", "subset_ids_str"],
            title="Accuracy vs n (all runs) — kernel color, mode symbol",
        )
        fig_scatter.update_yaxes(range=[0, 1])
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    else:
        fig_scatter = go.Figure()
        fig_scatter.update_layout(title="Accuracy vs n scatter — not available")

    # -----------------------------
    # Return as dict for flexible notebook display
    # -----------------------------
    return {
        "heatmap_kernel_mode": fig_heat,
        "acc_vs_n_by_mode": fig_n,
        "acc_vs_testsize_by_mode": fig_ts,
        "scatter_acc_vs_n": fig_scatter,
        "top_runs_table": fig_table,
    }

def fig2_style_svm_from_kernel(
    K: np.ndarray,
    y,
    *,
    class_a=None,
    class_b=None,
    n_points_grid: int = 250,
    C: float = 1.0,
    seed: int = 42,
    title: str = "Fig-2 style SVM view (KernelPCA → 2D)"
):
    """
    Fig-2-Style Plot:
    1) KernelPCA (precomputed kernel) -> 2D coords
    2) Train linear SVM on 2D coords (binary)
    3) Plot:
       - Left: points + support vectors + decision boundary
       - Right: grid colored by predicted class + boundary (decision=0)
    """

    y = np.asarray(y)

    # --- pick two classes automatically if not provided ---
    uniq, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)
    if class_a is None or class_b is None:
        if len(uniq) < 2:
            raise ValueError("Need at least 2 classes for binary Fig-2 plot.")
        class_a = uniq[order[0]]
        class_b = uniq[order[1]]

    mask = (y == class_a) | (y == class_b)
    K2 = K[np.ix_(mask, mask)]
    y2 = y[mask]
    ybin = np.where(y2 == class_a, 0, 1)

    # --- KernelPCA to 2D using the precomputed kernel matrix ---
    kpca = KernelPCA(n_components=2, kernel="precomputed", random_state=seed)
    X2d = kpca.fit_transform(K2)

    # --- Train linear SVM in the 2D projection space ---
    clf = SVC(kernel="linear", C=C, random_state=seed)
    clf.fit(X2d, ybin)

    # --- Build a grid over the 2D space ---
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

    # --- Support vectors in the 2D projection ---
    sv_idx = clf.support_
    sv = X2d[sv_idx]

    # -------- Plotly figure with two panels --------
    fig = go.Figure()

    # RIGHT panel: colored grid (predictions) + boundary line at decision=0
    fig.add_trace(
        go.Contour(
            x=gx, y=gy, z=pred.astype(float),
            showscale=False,
            contours=dict(coloring="fill"),
            opacity=0.35,
            hoverinfo="skip",
            name="Predicted regions"
        )
    )
    fig.add_trace(
        go.Contour(
            x=gx, y=gy, z=dec,
            contours=dict(start=0, end=0, size=1),
            showscale=False,
            line=dict(width=3),
            hoverinfo="skip",
            name="Decision boundary (0)"
        )
    )

    # LEFT overlay: points + support vectors
    # (Wir legen sie “oben drauf” – wirkt wie linker Plot im Paper.)
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

    return fig, {"class_a": class_a, "class_b": class_b, "n_samples": int(mask.sum())}

def plot_heatmaps_by_k(df, metric="accuracy"):
    df = df.dropna(subset=["k"])
    df["kernel_mode"] = df["kernel"] + " | " + df["mode"]

    ks = sorted(df["k"].unique())

    fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), sharey=True)

    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        pivot = (
            df[df["k"] == k]
            .pivot_table(
                index="kernel",
                columns="mode",
                values=metric,
                aggfunc="mean",
            )
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

def plot_bar_comparison(df, group_by="k", title_prefix=""):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ✅ WICHTIG: sofort kopieren
    d = df.copy()

    # Optionales Filtern
    if group_by in d.columns:
        d = d.dropna(subset=[group_by]).copy()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=d,
        x=group_by,
        y="accuracy",
        hue="kernel",
        ci="sd",
        palette="Set2"
    )

    plt.title(f"{title_prefix} (mean ± std)")
    plt.ylim(0.3, 1.0)
    plt.tight_layout()
    plt.show()


def plot_difference_heatmap(df):
    df = df.dropna(subset=["k"])

    pivot = (
        df.pivot_table(
            index=["k", "mode"],
            columns="kernel",
            values="accuracy",
            aggfunc="mean"
        )
        .reset_index()
    )

    pivot["diff"] = pivot["DRF–WL"] - pivot["ITS–WL"]

    diff_pivot = pivot.pivot(
        index="mode",
        columns="k",
        values="diff"
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        diff_pivot,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )
    plt.title("Accuracy difference (DRF − ITS)")
    plt.tight_layout()
    plt.show()

def plot_slope_drf_vs_its(df, k=None, tag=None):
    d = df.copy()
    if k is not None:
        d = d[d["k"] == k]
    if tag is not None:
        d = d[d["tag"] == tag]

    agg = (d.groupby(["kernel", "mode"], as_index=False)["accuracy"]
             .mean())

    # Pivot: rows=mode, cols=kernel
    p = agg.pivot(index="mode", columns="kernel", values="accuracy").reset_index()
    # Erwartete Spaltennamen
    if "DRF–WL" not in p.columns or "ITS–WL" not in p.columns:
        raise ValueError("Erwarte kernel-Spalten 'DRF–WL' und 'ITS–WL' in df['kernel'].")

    fig = go.Figure()
    for _, row in p.iterrows():
        fig.add_trace(go.Scatter(
            x=["ITS–WL", "DRF–WL"],
            y=[row["ITS–WL"], row["DRF–WL"]],
            mode="lines+markers+text",
            text=[None, row["mode"]],
            textposition="middle right",
            name=row["mode"],
            showlegend=False
        ))

    title = "DRF vs ITS (mean accuracy)"
    if tag: title += f" | {tag}"
    if k is not None: title += f" | k={k}"
    fig.update_layout(title=title, yaxis_title="accuracy", xaxis_title="")
    fig.show()

def plot_drf_minus_its_bar(df):
    import matplotlib.pyplot as plt

    d = df.copy()
    d = d[d["kernel"].isin(["DRF–WL", "ITS–WL"])]

    # DRF - ITS Difference vorbereiten
    pivot = (
        d.groupby(["mode", "k", "n", "test_size", "kernel"])["accuracy"]
        .mean()
        .unstack("kernel")
        .reset_index()
    )
    pivot["diff"] = pivot["DRF–WL"] - pivot["ITS–WL"]

    labels = pivot.apply(
        lambda r: f"{r['mode']} | k={int(r['k'])}", axis=1
    )

    plt.figure(figsize=(7, 4))
    plt.barh(labels, pivot["diff"], color="steelblue")
    plt.axvline(0, color="black", lw=1)

    plt.xlabel("Accuracy difference (DRF − ITS)")
    plt.title("DRF vs ITS performance difference")
    plt.tight_layout()
    plt.show()

def plot_drf_vs_its_dots(df):
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
        jitter=0.15
    )

    plt.title("DRF vs ITS accuracy distribution")
    plt.tight_layout()
    plt.show()

def plot_accuracy_by_k(df):
    import seaborn as sns
    import matplotlib.pyplot as plt

    d = df.dropna(subset=["k"]).copy()

    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=d,
        x="k",
        y="accuracy",
        hue="kernel",
        style="mode",
        markers=True,
        dashes=False
    )

    plt.title("Accuracy vs k (shared classes)")
    plt.tight_layout()
    plt.show()




# ============================================================
# WP3 Plots (Extended): k=1 vs k=2 comparisons + confidence bands
# ============================================================

# ------------------------------------------------------------
# Helper: aggregate mean/std across seeds
# ------------------------------------------------------------
def _agg_mean_std(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Aggregiert Runs (z.B. mehrere Seeds) zu mean/std pro Setting.
    Erwartet Spalten: metric + group_cols.
    """
    g = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_runs"})
    )
    g[f"{metric}_std"] = g[f"{metric}_std"].fillna(0.0)
    return g


# ------------------------------------------------------------
# Plot 1: Accuracy curves with confidence bands (mean ± std)
# ------------------------------------------------------------
def plot_acc_curves_with_bands(
    df_results: pd.DataFrame,
    *,
    x: str = "n",
    metric: str = "accuracy",
    facet_col: str = "mode",
    color_col: str = "kernel",
    line_dash_col: Optional[str] = "k",  # k=1 vs k=2
    title: str = "Accuracy vs n (mean ± std across seeds)",
) -> go.Figure:
    """
    Erwartet in df_results mindestens:
      - kernel, mode, n, accuracy
    Optional:
      - seed (für mehrere Runs)
      - k (z.B. 1 oder 2) um Linien zu trennen
    """
    df = df_results.copy()
    needed = {x, metric, facet_col, color_col}
    miss = needed - set(df.columns)
    if miss:
        raise KeyError(f"df_results missing columns: {sorted(miss)}")

    group_cols = [facet_col, color_col, x]
    if line_dash_col is not None and line_dash_col in df.columns:
        group_cols.insert(2, line_dash_col)  # kernel/mode/k/n

    agg = _agg_mean_std(df, group_cols=group_cols, metric=metric)

    # Wir bauen die Figur manuell (besser für Bands)
    fig = go.Figure()

    # Facets manuell: wir zeichnen pro mode in separaten "legendgroups" nicht,
    # sondern machen ein "pseudo-facet" über separate traces mit Annotationen.
    # (Einfacher + robust)
    modes = sorted(agg[facet_col].unique().tolist())

    # Layout: ein Plot pro mode (untereinander)
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=len(modes),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{facet_col}={m}" for m in modes],
        vertical_spacing=0.08,
    )

    # Traces
    for r, m in enumerate(modes, start=1):
        sub = agg[agg[facet_col] == m].copy()

        # Gruppen: kernel (+ optional k)
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

            # mean line
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

            # band: mean ± std
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


# ------------------------------------------------------------
# Plot 2: Heatmap "kernel vs mode" (mean accuracy)
# ------------------------------------------------------------
def plot_kernel_mode_heatmap(
    df_results: pd.DataFrame,
    *,
    metric: str = "accuracy",
    title: str = "Mean accuracy by kernel × mode",
    include_k: bool = True,
) -> go.Figure:
    """
    Gibt eine Heatmap zurück:
      y = kernel (optional mit k)
      x = mode
      z = mean accuracy
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
    )
    fig.update_layout(width=900, height=380 + 22 * len(pivot.index))
    return fig


# ------------------------------------------------------------
# Plot 3: DRF vs ITS difference heatmap
# ------------------------------------------------------------
def plot_drf_minus_its_difference_heatmap(
    df_results: pd.DataFrame,
    *,
    metric: str = "accuracy",
    group_cols: Optional[list[str]] = None,
    title: str = "Difference heatmap: DRF–WL minus ITS–WL (mean accuracy)",
) -> go.Figure:
    """
    Baut eine Differenz-Heatmap (DRF - ITS) über Settings.
    Default gruppiert nach: mode, n, test_size, k.
    Erwartet kernel-Spalte mit mindestens "DRF" und "ITS" in den Namen.
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

    # normalize kernel labels
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

    # pivot so we have DRF and ITS side-by-side
    piv = agg.pivot_table(index=group_cols, columns="kernel_kind", values="mean", fill_value=0.0).reset_index()
    if "DRF" not in piv.columns or "ITS" not in piv.columns:
        raise ValueError("Need both DRF and ITS runs in df_results for a difference heatmap.")

    piv["diff"] = piv["DRF"] - piv["ITS"]

    # Build a readable y-axis label per setting row
    def _row_label(row) -> str:
        parts = []
        for c in group_cols:
            parts.append(f"{c}={row[c]}")
        return " | ".join(parts)

    piv["setting"] = piv.apply(_row_label, axis=1)

    # For a heatmap, we use x = mode (if present) else x = "diff"
    # Better: x = n if present, else mode.
    if "n" in piv.columns:
        xcol = "n"
    elif "mode" in piv.columns:
        xcol = "mode"
    else:
        xcol = group_cols[0]

    # If multiple modes exist, separate rows by mode
    # We'll do: y = setting, x = (xcol), z = diff
    mat = piv.pivot_table(index="setting", columns=xcol, values="diff", fill_value=0.0)

    fig = px.imshow(
        mat.values,
        x=list(mat.columns),
        y=list(mat.index),
        aspect="auto",
        title=title,
    )
    fig.update_layout(width=950, height=420 + 20 * len(mat.index))
    return fig


# ------------------------------------------------------------
# Dashboard: return all figs in one dict (stable keys)
# ------------------------------------------------------------
def plot_experiment_dashboard(
    df_results: pd.DataFrame,
    *,
    title_prefix: str = "WP3 Dashboard",
) -> Dict[str, go.Figure]:
    """
    Gibt ein dict mit stabilen Keys zurück:
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