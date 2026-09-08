"""Shared numerical helpers used across the gRely featurization pipeline.

IMPORTANT — provenance and behavior preservation
-------------------------------------------------
Every function in this module is a **verbatim copy** of the corresponding
function in the original pipeline scripts (primarily
``scripts/04_featurization/compile_final_feature_set.py``). It is provided so
that the shared logic has a single importable home.

The pipeline scripts under ``scripts/`` still contain their own inline copies
of these functions and have deliberately been left untouched, because the
restructuring could not be executed/verified in this environment and the
project's rule is that any change in numerical output is a bug, not an
improvement. Wiring the scripts to import from here is a recommended *next*
step that must be done against a live environment and validated for numerical
equivalence (see ``MIGRATION.md`` → "Unverified / next steps").

Do NOT "unify" this module with ``extract_celltype_features``: that function
diverges between ``gene_data_features.py`` (uses ``scipy.stats.variation``) and
``compile_final_feature_set.py`` (uses a stabilized ``nanstd/nanmean``). That
divergence is intentional-looking and is documented in ``MIGRATION.md``; it is
therefore *not* centralized here.
"""

import numpy as np
import re


def nanpercentile(a, q):
    finite_vals = a[np.isfinite(a)]
    return np.percentile(finite_vals, q) if finite_vals.size > 0 else np.nan


def summarize_vector(v, prefix: str):
    v = np.asarray(v, dtype=float)
    n = v.size
    av = np.abs(v)
    s1 = float(av.sum())

    feats = {}

    # Moments / norms / range
    feats[f"{prefix}_mean"]   = float(v.mean())
    feats[f"{prefix}_std"]    = float(v.std(ddof=0))
    feats[f"{prefix}_median"] = float(np.median(v))
    feats[f"{prefix}_mad"]    = float(1.4826 * np.median(np.abs(v - np.median(v))))  # robust scale
    feats[f"{prefix}_l1"]     = float(av.sum())
    feats[f"{prefix}_l2"]     = float(np.sqrt(np.dot(v, v)))
    feats[f"{prefix}_linf"]   = float(av.max(initial=0.0))
    feats[f"{prefix}_min"]    = float(v.min(initial=0.0))
    feats[f"{prefix}_max"]    = float(v.max(initial=0.0))

    # Quantiles (robust summaries)
    qs = np.quantile(v, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    for qv, qn in zip(qs, ("q01","q05","q25","q50","q75","q95","q99")):
        feats[f"{prefix}_{qn}"] = float(qv)

    # Sign structure
    pos = v[v > 0]; neg = v[v < 0]
    feats[f"{prefix}_frac_pos"] = float(pos.size / n)
    feats[f"{prefix}_frac_neg"] = float(neg.size / n)
    feats[f"{prefix}_mean_pos"] = float(pos.mean()) if pos.size else 0.0
    feats[f"{prefix}_mean_neg"] = float(neg.mean()) if neg.size else 0.0
    feats[f"{prefix}_l1_pos"]   = float(pos.sum()) if pos.size else 0.0
    feats[f"{prefix}_l1_neg"]   = float((-neg).sum()) if neg.size else 0.0  # absolute of negatives

    # Near-zero sparsity at multiple thresholds
    for eps in (1e-8, 1e-6, 1e-4):
        feats[f"{prefix}_frac_abs_lt_{eps:g}"] = float((av < eps).mean())

    # Gini coefficient (concentration of |v| across tracks)
    if s1 > 0.0:
        y_sorted = np.sort(av)
        i = np.arange(1, n + 1)
        gini = 1.0 + 1.0/n - 2.0 * np.sum((n + 1 - i) * y_sorted) / (n * s1)
    else:
        gini = 0.0
    feats[f"{prefix}_gini_abs"] = float(gini)

    # Entropy of |v| mass (base-2); higher = more diffuse across tracks
    if s1 > 0.0:
        p = av / s1
        nz = p > 0
        H = -float(np.sum(p[nz] * np.log2(p[nz])))
    else:
        H = 0.0
    feats[f"{prefix}_entropy_abs_bits"] = H

    # Tail mass concentration: fraction of |v| in top k% tracks
    for frac in (0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(frac * n)))
        if s1 > 0.0:
            idx = np.argpartition(av, -k)[-k:]
            feats[f"{prefix}_top{int(frac*100)}pct_mass_abs"] = float(av[idx].sum() / s1)
        else:
            feats[f"{prefix}_top{int(frac*100)}pct_mass_abs"] = 0.0

    return feats


def load_vector_nan_safe(path):
    """Load a numpy vector, convert inf to nan, keep dtype float."""
    arr = np.asarray(np.load(path), dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan  # turn ±inf into NaN
    return arr


def get_phylop_window_mean(bw_obj, chrom, center_pos, window_size, neutral_score=0.0):
    """Return mean phyloP score in ±window_bp region, replacing missing with neutral_score."""
    flank_bp = window_size//2
    start = max(center_pos - flank_bp - 1, 0)  # 0-based inclusive
    end = center_pos + flank_bp               # 1-based exclusive
    vals = bw_obj.values(chrom, start, end)
    clean_vals = [
        v if (v is not None and not (isinstance(v, float) and v != v)) else neutral_score
        for v in vals
    ]
    return np.mean(clean_vals)


def compute_tau(x):
    """Tissue-specificity index τ from Yanai et al., 2005"""
    if np.max(x) == 0:
        return 0.0
    x_norm = x / np.max(x)
    return np.sum(1 - x_norm) / (len(x_norm) - 1)


def compute_gini(x):
    """Compute Gini coefficient"""
    if np.amin(x) < 0:
        x = x - np.amin(x)
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x) + 1e-8)


def _slug(x):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(x).strip())


def compute_specificity_metrics(x, i, epsilon=1e-8):
    others = np.delete(x, i)
    z = (x[i] - np.mean(others)) / (np.std(others) + epsilon)
    fc = (x[i] + epsilon) / (np.mean(others) + epsilon)
    frac = x[i] / (np.sum(x) + epsilon)
    rel_expr = others / (x[i] + epsilon)
    spec = 1 - np.mean(rel_expr)
    return z, fc, frac, spec
