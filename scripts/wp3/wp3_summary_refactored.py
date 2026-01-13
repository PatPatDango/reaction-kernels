import logging
from pathlib import Path

import numpy as np

from wp3_core import (
    load_precomputed_features,
    kernel_multiset_intersection,
    compute_kernel_matrix,
    kernel_matrix_stats,
    find_first_nonzero_pair,
    train_svm_with_precomputed_kernel,
    available_subset_ids,
    ResultsCollector,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def find_repo_root() -> Path:
    """
    Versucht zuerst Git-Root; fällt zurück auf Ordner mit 'data' oder 'scripts'.
    """
    cwd = Path.cwd()
    # git root
    try:
        import subprocess

        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=cwd).decode().strip()
        root_p = Path(root)
        if root_p.exists():
            return root_p
    except Exception:
        pass

    # heuristik
    cand = next((p for p in [cwd, *cwd.parents] if (p / "data").is_dir() or (p / "scripts").is_dir()), None)
    if cand is None:
        raise RuntimeError("Repo-Root not found (expected folder 'data' or 'scripts').")
    return cand


def main():
    ROOT = find_repo_root()
    DATA_DIR = ROOT / "data"
    log.info("ROOT: %s", ROOT)
    log.info("DATA_DIR: %s", DATA_DIR)

    # Verzeichnisdefinitionen
    DRF_DIRS = {
        "edge": DATA_DIR / "drf_small" / "precomputed_drf_edge",
        "vertex": DATA_DIR / "drf_small" / "precomputed_drf_vertex",
        "sp": DATA_DIR / "drf_small" / "precomputed_drf_sp",
    }
    ITS_DIRS = {
        "edge": DATA_DIR / "its_small" / "precomputed_its_edge",
        "vertex": DATA_DIR / "its_small" / "precomputed_its_vertex",
        "sp": DATA_DIR / "its_small" / "precomputed_its_sp",
    }

    for k, v in DRF_DIRS.items():
        log.info("DRF %-6s -> %s", k, v)
    for k, v in ITS_DIRS.items():
        log.info("ITS %-6s -> %s", k, v)

    # Laden nach Mode
    X_drf, y_drf = {}, {}
    X_its, y_its = {}, {}

    for mode, p in DRF_DIRS.items():
        assert p.exists(), f"Pfad nicht gefunden: {p}"
        X, y = load_precomputed_features(p, feature_key="drf_wl")
        X_drf[mode], y_drf[mode] = X, y
        log.info("Loaded DRF (%s) | n=%d | classes=%d", mode, len(X), len(set(y)))

    for mode, p in ITS_DIRS.items():
        assert p.exists(), f"Pfad nicht gefunden: {p}"
        X, y = load_precomputed_features(p, feature_key="its_wl")
        X_its[mode], y_its[mode] = X, y
        log.info("Loaded ITS (%s) | n=%d | classes=%d", mode, len(X), len(set(y)))

    # Beispiel: erstes Paar mit Kernel>0 suchen
    mode = "edge"
    pair = find_first_nonzero_pair(X_its[mode], kernel_multiset_intersection, min_overlap=1)
    if pair:
        i, j, k = pair
        log.info("First non-zero kernel (ITS-%s): i=%d j=%d value=%d", mode, i, j, k)
    else:
        log.info("No non-zero pair found in ITS-%s", mode)

    # Kernel-Matrix-Stats (DRF/ITS)
    n = 200
    K_drf = compute_kernel_matrix(X_drf[mode][:n])
    K_its = compute_kernel_matrix(X_its[mode][:n])

    stats_drf = kernel_matrix_stats(K_drf)
    stats_its = kernel_matrix_stats(K_its)
    log.info("Kernel stats DRF-%s (n=%d): %s", mode, n, stats_drf)
    log.info("Kernel stats ITS-%s (n=%d): %s", mode, n, stats_its)

    # Baseline-Training: gemeinsame Subsets (edge)
    drf_ids = set(available_subset_ids(DRF_DIRS["edge"]))
    its_ids = set(available_subset_ids(ITS_DIRS["edge"]))
    common_ids = sorted(drf_ids & its_ids)
    if not common_ids:
        raise FileNotFoundError("Keine gemeinsamen subset_XXX PKLs zwischen DRF(edge) und ITS(edge).")
    subset_ids = common_ids[:10]
    log.info("Chosen subset_ids: %s", subset_ids)

    # SVM-Training (DRF/ITS edge)
    C = 1.0
    seed = 42
    test_size = 0.2
    n_train = 600

    rc = ResultsCollector()

    # DRF edge
    res_drf = train_svm_with_precomputed_kernel(
        X_drf["edge"], y_drf["edge"],
        n=n_train, test_size=test_size, seed=seed, C=C, normalize=True, verbose=True
    )
    rc.add(
        tag="baseline",
        representation="DRF–WL",
        mode="edge",
        n=n_train,
        test_size=test_size,
        C=C,
        seed=seed,
        acc=res_drf.acc,
        subset_ids=subset_ids,
        feature_key="drf_wl",
    )

    # ITS edge
    res_its = train_svm_with_precomputed_kernel(
        X_its["edge"], y_its["edge"],
        n=n_train, test_size=test_size, seed=seed, C=C, normalize=True, verbose=True
    )
    rc.add(
        tag="baseline",
        representation="ITS–WL",
        mode="edge",
        n=n_train,
        test_size=test_size,
        C=C,
        seed=seed,
        acc=res_its.acc,
        subset_ids=subset_ids,
        feature_key="its_wl",
    )

    # Zusammenfassung
    df = rc.to_frame()
    # Kleiner Überblick
    cols = ["tag", "representation", "mode", "n", "test_size", "accuracy", "subset_ids"]
    with np.printoptions(precision=3, suppress=True):
        log.info("\n%s", df[cols].sort_values(["tag", "representation", "mode"]).to_string(index=False))

    # Optional speichern
    out = ROOT / "results" / "wp3_summary.csv"
    rc.save_csv(out)
    log.info("Saved summary to %s", out)


if __name__ == "__main__":
    main()