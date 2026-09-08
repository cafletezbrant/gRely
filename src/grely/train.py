"""Train the gRely meta-model (XGBoost) with leave-one-tissue-set-and-chr1-3-out CV.

The model, hyperparameters, per-fold ``scale_pos_weight = n_neg / n_pos``, and
random seed are defined in ``configs/grely.yaml`` → ``meta_model``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import (
    get_chroms,
    get_folds,
    get_label,
    load_config,
    split_masks,
    union,
)


def make_model(scale_pos_weight: float, config: dict | None = None):
    """Construct the gRely ``XGBClassifier`` from the configured hyperparameters."""
    from xgboost import XGBClassifier  # imported lazily so the package imports without xgboost

    mm = (config or load_config())["meta_model"]
    return XGBClassifier(
        n_estimators=int(mm["n_estimators"]),
        max_depth=int(mm["max_depth"]),
        learning_rate=float(mm["learning_rate"]),
        subsample=float(mm["subsample"]),
        colsample_bytree=float(mm["colsample_bytree"]),
        use_label_encoder=bool(mm.get("use_label_encoder", False)),
        eval_metric=mm.get("eval_metric", "logloss"),
        scale_pos_weight=scale_pos_weight,
        random_state=int(mm["random_state"]),
    )


def _spw(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def train_fold(
    df: pd.DataFrame,
    features: list[str],
    fold_name: str,
    out_models: str = "models",
    config: dict | None = None,
):
    """Train one fold: hold out ``fold_name``'s tissues + test chroms, train on the
    rest, and save ``{out_models}/xgb_model_center_196kb_{fold_name}_chromosomal.json``.

    Returns ``(model, train_mask, test_mask)``.
    """
    cfg = config or load_config()
    folds = get_folds(cfg)
    train_chroms, test_chroms = get_chroms(cfg)

    fold_map = dict(folds)
    test_tissues = fold_map[fold_name]
    train_tissues = union(*[s for (n, s) in folds if n != fold_name])

    train_mask, test_mask = split_masks(df, test_tissues, train_tissues, train_chroms, test_chroms)
    y = get_label(df)
    X_train = df.loc[train_mask, features]
    y_train = y.loc[train_mask]

    model = make_model(_spw(y_train), cfg)
    model.fit(X_train, y_train)

    os.makedirs(out_models, exist_ok=True)
    model_path = os.path.join(out_models, f"xgb_model_center_196kb_{fold_name}_chromosomal.json")
    model.save_model(model_path)
    print(f"[{fold_name}] train={int(train_mask.sum())} test={int(test_mask.sum())} -> {model_path}")
    return model, train_mask, test_mask


def train_all_folds(
    df: pd.DataFrame,
    features: list[str],
    out_models: str = "models",
    out_prob: str | None = "yprob",
    config: dict | None = None,
):
    """Train every fold and, optionally, save held-out predicted probabilities to
    ``{out_prob}/yprob_{fold_name}.npy`` (the held-out fold model scores its own
    test tuples)."""
    cfg = config or load_config()
    y = get_label(df)
    for fold_name, _ in get_folds(cfg):
        model, _, test_mask = train_fold(df, features, fold_name, out_models, cfg)
        if out_prob:
            os.makedirs(out_prob, exist_ok=True)
            y_prob = model.predict_proba(df.loc[test_mask, features])[:, 1]
            np.save(os.path.join(out_prob, f"yprob_{fold_name}.npy"), y_prob)
            print(f"  saved {out_prob}/yprob_{fold_name}.npy (n={len(y_prob)})")
