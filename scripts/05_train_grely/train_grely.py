#!/usr/bin/env python
"""Train gRely on the compiled feature matrix.

Stage 05 of the pipeline. Reads the compiled (variant, gene, tissue) feature
matrix produced by stage 04, prepares features/labels/folds, and trains one
XGBoost model per cross-validation fold (hold out a tissue set + chr1-3).

Example
-------
    python scripts/05_train_grely/train_grely.py \
        --matrix data/task_sign_matching_with_features.parquet \
        --out-models models --out-prob yprob

Model, hyperparameters, and folds come from configs/grely.yaml (values identical
to the paper). Use --fold to train a single fold.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make `grely` importable when run from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grely import data as gdata          # noqa: E402
from grely import train as gtrain        # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train the gRely meta-model.")
    ap.add_argument("--matrix", required=True,
                    help="Compiled feature matrix parquet (stage 04 output, with labels).")
    ap.add_argument("--out-models", default="models", help="Directory for saved fold models.")
    ap.add_argument("--out-prob", default="yprob",
                    help="Directory for held-out probabilities (pass '' to skip).")
    ap.add_argument("--fold", default=None,
                    help="Train only this fold (e.g. set5). Default: all folds.")
    ap.add_argument("--config", default=None, help="Path to grely.yaml (default: configs/grely.yaml).")
    args = ap.parse_args()

    cfg = gdata.load_config(args.config)
    df = pd.read_parquet(args.matrix)
    df, features, _ = gdata.prepare(df, cfg)
    print(f"Loaded matrix: {df.shape[0]} tuples, {len(features)} features.")

    if args.fold:
        gtrain.train_fold(df, features, args.fold, out_models=args.out_models, config=cfg)
    else:
        gtrain.train_all_folds(
            df, features,
            out_models=args.out_models,
            out_prob=(args.out_prob or None),
            config=cfg,
        )


if __name__ == "__main__":
    main()
