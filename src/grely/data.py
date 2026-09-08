"""Feature-matrix preparation and cross-validation splits for gRely.

This is a faithful extraction of the data-preparation logic that lived in the
original training/evaluation notebooks (feature selection, the center-window
filter, the label definition, and the joint tissue+chromosome fold masks). The
numbers it produces are unchanged from the notebook; only the packaging is new.

The single source of truth for the fold/tissue/feature definitions is
``configs/grely.yaml`` — this module reads it so the config and code cannot
drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRELY_CONFIG = REPO_ROOT / "configs" / "grely.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load ``configs/grely.yaml`` (or ``$GRELY_MODEL_CONFIG``)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required — `pip install pyyaml`.")
    cfg = config_path or os.environ.get("GRELY_MODEL_CONFIG") or DEFAULT_GRELY_CONFIG
    with open(cfg) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Feature / label selection  (verbatim logic from the original notebooks)
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, config: dict | None = None) -> list[str]:
    """Return the ordered gRely feature columns.

    Mirrors the notebook exactly::

        features = list(df.columns[first_col:]) + [magnitude_feature]
        features = [f for f in features if f not in leakage_columns]

    The first ``first_col`` columns are identifier/label/metadata columns; the
    ``leakage_columns`` (eQTL-summary-statistic-derived) are dropped to prevent
    label leakage, per the paper's Methods.
    """
    cfg = (config or load_config()).get("features", {})
    first_col = int(cfg.get("first_feature_col_index", 40))
    magnitude = cfg.get("magnitude_feature", "avg_logSum")
    leakage = set(cfg.get("leakage_columns", []))

    features = list(df.columns[first_col:]) + [magnitude]
    features = [f for f in features if f not in leakage]
    return features


def add_chrom(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``chrom`` column derived from the variant id (``chrN_pos_ref_alt``)."""
    df = df.copy()
    df["chrom"] = [x.split("_")[0] for x in df["variant"]]
    return df


def filter_center_window(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Keep tuples whose target gene lies in the central window.

    ``variant_to_tss_distance < center_window_bp`` (default 98,304 = 196,608//2).
    """
    cfg = (config or load_config()).get("cross_validation", {})
    window = int(cfg.get("center_window_bp", 196608 // 2))
    return df[df["variant_to_tss_distance"] < window].copy()


def get_label(df: pd.DataFrame) -> pd.Series:
    """Binary sign-concordance label: ``logSum_sign_correct`` -> {1, 0}."""
    return df["logSum_sign_correct"].map({True: 1, False: 0})


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------

def get_folds(config: dict | None = None) -> list[tuple[str, list[str]]]:
    """Return ``[(fold_name, [tissues...]), ...]`` from the config tissue sets."""
    sets = (config or load_config())["tissue_sets"]
    return [(name, list(sets[name])) for name in sorted(sets)]


def get_chroms(config: dict | None = None) -> tuple[list[str], list[str]]:
    """Return ``(train_chroms, test_chroms)`` from the config."""
    cv = (config or load_config())["cross_validation"]
    return list(cv["train_chroms"]), list(cv["test_chroms"])


def union(*lists) -> set:
    out: set = set()
    for L in lists:
        out.update(L)
    return out


def split_masks(df, test_tissues, train_tissues, train_chroms, test_chroms):
    """Joint tissue+chromosome train/test masks (verbatim from the notebook)."""
    train_mask = df["chrom"].isin(train_chroms) & df["target_tissue"].isin(train_tissues)
    test_mask = df["chrom"].isin(test_chroms) & df["target_tissue"].isin(test_tissues)
    return train_mask, test_mask


def prepare(df: pd.DataFrame, config: dict | None = None):
    """Convenience: add chrom, apply the center-window filter, and return
    ``(df, features, label)`` ready for training/scoring."""
    cfg = config or load_config()
    df = filter_center_window(add_chrom(df), cfg)
    features = select_features(df, cfg)
    y = get_label(df)
    return df, features, y
