"""Evaluate gRely: coverage vs. accuracy, magnitude baseline, and per-tissue reliability.

Given a pooled set of held-out predictions (each scored by the fold model that
held it out — see `grely.apply.predict_out_of_fold`), this computes the metrics
behind the paper's reliability claims, as source-data tables (no plotting):

- `coverage_curve` — sign-concordance accuracy and gene coverage as gRely (or
  effect-size) thresholds retain fewer predictions.
- `precision_recall` / `average_precision_by_fold` — the sign-concordance task.
- `threshold_table` — magnitude thresholding vs. gRely quantile filtering,
  including the low-effect regime that magnitude filtering discards.
- `per_tissue_quantile_accuracy` — baseline / top / bottom quantile accuracy per
  tissue.
- `effect_size_correlation` — predicted VEP vs. observed eQTL effect size.
- `mean_grely_per_tissue` — mean gRely score vs. observed concordance per tissue.

Uses only numpy / pandas / scipy (all core dependencies).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Default label / column names in the compiled feature matrix.
LABEL_COL = "logSum_sign_correct"
SCORE_COL = "grely_prob"
MAG_COL = "avg_logSum"          # signed VEP; magnitude is its absolute value
BETA_COL = "beta"
TISSUE_COL = "target_tissue"
GENE_COL = "gene_id"


def _acc(labels: pd.Series, mask) -> float:
    m = np.asarray(mask)
    return float(labels[m].astype(float).mean()) if m.sum() > 0 else float("nan")


# ---------------------------------------------------------------------------
# Coverage vs. accuracy (and gene coverage)
# ---------------------------------------------------------------------------

def coverage_curve(df: pd.DataFrame, rank_col: str = SCORE_COL, label_col: str = LABEL_COL,
                   gene_col: str = GENE_COL, abs_rank: bool = False,
                   quantiles=None) -> pd.DataFrame:
    """Sweep thresholds on ``rank_col``; report coverage, accuracy, gene coverage.

    At each threshold (a quantile of ``rank_col``), predictions with
    ``rank_col >= threshold`` are retained. With ``abs_rank=True`` the ranking
    uses ``|rank_col|`` — use that for the effect-size (``avg_logSum``) baseline.
    """
    if quantiles is None:
        quantiles = np.linspace(0.0, 1.0, 21)          # 0%, 5%, ..., 100%
    rank = df[rank_col].abs() if abs_rank else df[rank_col]
    labels = df[label_col]
    n = len(df)
    n_genes = df[gene_col].nunique() if gene_col in df.columns else np.nan

    rows = []
    for q in quantiles:
        thr = float(rank.quantile(q))
        keep = rank >= thr
        gene_cov = (df.loc[keep, gene_col].nunique() / n_genes
                    if gene_col in df.columns and n_genes else np.nan)
        rows.append({
            "quantile": float(q),
            "threshold": thr,
            "coverage": float(keep.mean()),
            "n_retained": int(keep.sum()),
            "accuracy": _acc(labels, keep),
            "gene_coverage": float(gene_cov) if gene_cov == gene_cov else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Precision / recall on the sign-concordance task
# ---------------------------------------------------------------------------

def precision_recall(df: pd.DataFrame, score_col: str = SCORE_COL,
                     label_col: str = LABEL_COL) -> dict:
    """Precision-recall curve and average precision (dependency-free)."""
    y = df[label_col].astype(int).to_numpy()
    s = df[score_col].to_numpy()
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum() if y.sum() > 0 else np.zeros_like(tp, dtype=float)

    rec = np.concatenate([[0.0], recall])
    prec = np.concatenate([[1.0], precision])
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    return {"precision": precision, "recall": recall, "scores": s, "average_precision": ap}


def average_precision_by_fold(df: pd.DataFrame, score_col: str = SCORE_COL,
                              label_col: str = LABEL_COL, fold_col: str = "fold") -> pd.DataFrame:
    """Average precision per fold (returns one row per fold plus mean / std)."""
    rows = []
    for fold, g in df.groupby(fold_col):
        rows.append({"fold": fold, "n": len(g),
                     "average_precision": precision_recall(g, score_col, label_col)["average_precision"]})
    out = pd.DataFrame(rows)
    summary = pd.DataFrame([{"fold": "mean", "n": int(out["n"].sum()),
                             "average_precision": float(out["average_precision"].mean())},
                            {"fold": "std", "n": np.nan,
                             "average_precision": float(out["average_precision"].std(ddof=0))}])
    return pd.concat([out, summary], ignore_index=True)


# ---------------------------------------------------------------------------
# Magnitude thresholding vs. gRely quantile filtering
# ---------------------------------------------------------------------------

def threshold_table(df: pd.DataFrame, score_col: str = SCORE_COL, mag_col: str = MAG_COL,
                    label_col: str = LABEL_COL, mag_cutoffs=(0.1, 0.5, 0.9, 2.0),
                    grely_top=(0.5, 0.2), low_vep: float = 0.2) -> pd.DataFrame:
    """Compare effect-size (|VEP|) filtering to gRely quantile filtering.

    Rows: an unfiltered baseline; magnitude thresholds ``|VEP| > c``; gRely top
    quantiles; and, in the low-effect regime (``|VEP| < low_vep``), the accuracy
    gRely recovers where magnitude filtering offers no guidance.
    """
    n = len(df)
    mag = df[mag_col].abs()
    labels = df[label_col]
    grely_thresholds = {q: float(df[score_col].quantile(1.0 - q)) for q in grely_top}

    rows = [{"filter": "All", "n_retained": n, "frac_of_test": 1.0,
             "accuracy": _acc(labels, np.ones(n, dtype=bool))}]

    for c in mag_cutoffs:
        m = (mag > c).to_numpy()
        rows.append({"filter": f"|VEP| > {c}", "n_retained": int(m.sum()),
                     "frac_of_test": float(m.mean()), "accuracy": _acc(labels, m)})

    for q in grely_top:
        m = (df[score_col] >= grely_thresholds[q]).to_numpy()
        rows.append({"filter": f"gRely top {int(q * 100)}%", "n_retained": int(m.sum()),
                     "frac_of_test": float(m.mean()), "accuracy": _acc(labels, m)})

    bands = [("|VEP| < 0.1", (mag < 0.1)),
             (f"0.1 < |VEP| < {low_vep}", (mag >= 0.1) & (mag < low_vep))]
    for name, band in bands:
        bm = band.to_numpy()
        rows.append({"filter": name, "n_retained": int(bm.sum()),
                     "frac_of_test": float(bm.mean()), "accuracy": _acc(labels, bm)})
        for q in grely_top:
            m = (band & (df[score_col] >= grely_thresholds[q])).to_numpy()
            rows.append({"filter": f"{name} & gRely top {int(q * 100)}%",
                         "n_retained": int(m.sum()), "frac_of_test": float(m.mean()),
                         "accuracy": _acc(labels, m)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-tissue reliability
# ---------------------------------------------------------------------------

def per_tissue_quantile_accuracy(df: pd.DataFrame, score_col: str = SCORE_COL,
                                 label_col: str = LABEL_COL, tissue_col: str = TISSUE_COL,
                                 top_frac: float = 0.2, bottom_frac: float = 0.2) -> pd.DataFrame:
    """Baseline / top-quantile / bottom-quantile sign concordance per tissue.

    Quantile thresholds are global (across all pooled predictions), then applied
    within each tissue — as in the paper's per-tissue panels.
    """
    thr_top = float(df[score_col].quantile(1.0 - top_frac))
    thr_bot = float(df[score_col].quantile(bottom_frac))
    rows = []
    for tissue, g in df.groupby(tissue_col):
        lab = g[label_col]
        rows.append({
            tissue_col: tissue,
            "n": len(g),
            "baseline": _acc(lab, np.ones(len(g), dtype=bool)),
            "top": _acc(lab, (g[score_col] >= thr_top).to_numpy()),
            "bottom": _acc(lab, (g[score_col] <= thr_bot).to_numpy()),
        })
    out = pd.DataFrame(rows).sort_values("baseline").reset_index(drop=True)
    overall = {tissue_col: "ALL", "n": len(df),
               "baseline": _acc(df[label_col], np.ones(len(df), dtype=bool)),
               "top": _acc(df[label_col], (df[score_col] >= thr_top).to_numpy()),
               "bottom": _acc(df[label_col], (df[score_col] <= thr_bot).to_numpy())}
    return pd.concat([pd.DataFrame([overall]), out], ignore_index=True)


def effect_size_correlation(df: pd.DataFrame, mag_col: str = MAG_COL, beta_col: str = BETA_COL,
                            score_col: str = SCORE_COL, top_frac: float = 0.2) -> dict:
    """Spearman correlation of predicted VEP vs. observed eQTL effect size.

    Reported for all predictions and for the top ``top_frac`` by gRely score.
    """
    from scipy.stats import spearmanr

    def rho(sub):
        if len(sub) < 3:
            return float("nan")
        return float(spearmanr(sub[mag_col], sub[beta_col]).correlation)

    thr = float(df[score_col].quantile(1.0 - top_frac))
    top = df[df[score_col] >= thr]
    return {"spearman_all": rho(df), "n_all": len(df),
            "spearman_top": rho(top), "n_top": len(top), "top_frac": top_frac}


def mean_grely_per_tissue(df: pd.DataFrame, score_col: str = SCORE_COL,
                          label_col: str = LABEL_COL, tissue_col: str = TISSUE_COL):
    """Per-tissue mean gRely score vs. observed sign concordance (+ Spearman ρ)."""
    from scipy.stats import spearmanr

    g = (df.groupby(tissue_col)
           .agg(mean_grely=(score_col, "mean"),
                sign_concordance=(label_col, lambda x: float(x.astype(float).mean())),
                n=(score_col, "size"))
           .reset_index())
    rho = float(spearmanr(g["mean_grely"], g["sign_concordance"]).correlation) if len(g) >= 3 else float("nan")
    return g, rho
