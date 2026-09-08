"""Apply a trained gRely model to score predictions — including another model's VEPs.

Because gRely is trained on Borzoi features but parameterizes genes/tissues by
continuous properties (never categorical identity), the Borzoi-trained fold
models score any (variant, gene, tissue) tuple for which the same feature vector
is available. This module loads those fold models and (a) attaches a reliability
probability to a feature table and (b) reproduces the cross-model transfer
evaluation (top-20% vs bottom-20% sign concordance) used for AlphaGenome/Enformer.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import get_chroms, get_folds, load_config, split_masks, union


def load_model(model_path: str):
    """Load a saved gRely XGBoost model from JSON."""
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(model_path)
    return model


def score(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return ``P(prediction correct)`` for every row of ``df``."""
    return model.predict_proba(df[features])[:, 1]


def score_by_fold(
    df: pd.DataFrame,
    features: list[str],
    model_dir: str = "models",
    prob_col: str = "grely_prob",
    config: dict | None = None,
) -> pd.DataFrame:
    """Attach ``prob_col`` using the correct held-out model per test tuple.

    For each fold, the model that held out that fold's tissues + test chromosomes
    scores exactly that fold's test tuples (chr1-3 ∩ held-out tissues). Tuples
    outside every fold's test region are left as NaN.
    """
    cfg = config or load_config()
    folds = get_folds(cfg)
    train_chroms, test_chroms = get_chroms(cfg)

    out = df.copy()
    out[prob_col] = np.nan
    for fold_name, test_tissues in folds:
        train_tissues = union(*[s for (n, s) in folds if n != fold_name])
        _, test_mask = split_masks(out, test_tissues, train_tissues, train_chroms, test_chroms)
        if not test_mask.any():
            continue
        model_path = os.path.join(model_dir, f"xgb_model_center_196kb_{fold_name}_chromosomal.json")
        model = load_model(model_path)
        out.loc[test_mask, prob_col] = score(model, out.loc[test_mask], features)
    return out


def transfer_eval(
    df: pd.DataFrame,
    sign_col: str,
    prob_col: str = "grely_prob",
    top_pct: float = 80.0,
    bottom_pct: float = 20.0,
) -> dict:
    """Cross-model transfer evaluation (verbatim logic from the Enformer analysis).

    Compares overall sign concordance against the top / bottom quantiles ranked
    by the gRely reliability score.

    ``sign_col`` is a boolean column of whether the *other* model's VEP is sign
    concordant with the eQTL (e.g. ``enformer_sign_concordant`` /
    ``alphagenome_sign_concordant``).
    """
    d = df.dropna(subset=[prob_col]).copy()
    top_threshold = np.percentile(d[prob_col], top_pct)
    bot_threshold = np.percentile(d[prob_col], bottom_pct)

    df_top = d[d[prob_col] >= top_threshold]
    df_bot = d[d[prob_col] <= bot_threshold]

    result = {
        "n": int(len(d)),
        "overall_acc": float(d[sign_col].mean()),
        "top_threshold": float(top_threshold),
        "bot_threshold": float(bot_threshold),
        "top_acc": float(df_top[sign_col].mean()),
        "bot_acc": float(df_bot[sign_col].mean()),
        "n_top": int(len(df_top)),
        "n_bot": int(len(df_bot)),
        "per_tissue_overall": d.groupby("target_tissue")[sign_col].mean().sort_values(),
        "per_tissue_top": df_top.groupby("target_tissue")[sign_col].mean(),
        "per_tissue_bot": df_bot.groupby("target_tissue")[sign_col].mean(),
    }
    return result
