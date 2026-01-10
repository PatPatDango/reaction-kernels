import pandas as pd
import plotly.express as px

def plot_experiment_results(df: pd.DataFrame):
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