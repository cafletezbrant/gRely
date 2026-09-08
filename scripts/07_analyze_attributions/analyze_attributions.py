#!/usr/bin/env python
"""Analyze gRely feature attributions.

Stage 07 of the pipeline. Given the compiled feature matrix and the trained fold
models, this computes per-prediction SHAP attributions, embeds and clusters the
predictions from their attribution vectors (UMAP + KMeans), ranks feature
importance globally and by family, and quantifies which feature families drive
low-confidence (failure) predictions.

Outputs (written to --out-dir):
  global_importance.csv    per-feature mean |SHAP|
  family_importance.csv    mean |SHAP| aggregated by feature family
  embedding.csv            UMAP-1/2 + grely_prob + sign concordance + cluster + ids
  cluster_summary.csv      per-cluster size, mean score, concordance, top features
  failure_enrichment.csv   feature-family enrichment among low-confidence predictions
  embedding_by_*.png       simple UMAP scatters (unless --no-plot)

Example
-------
    python scripts/07_analyze_attributions/analyze_attributions.py \
        --matrix data/task_sign_matching_with_features.parquet \
        --model-dir models --out-dir analysis/attribution --n-clusters 14

Requires the interpretation extras: pip install -e ".[interpret]"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from grely import attribution as gattr   # noqa: E402
from grely import data as gdata           # noqa: E402
from grely.paths import load_paths        # noqa: E402


def main():
    PATHS = load_paths()
    ap = argparse.ArgumentParser(description="Analyze gRely feature attributions (SHAP / UMAP / clusters).")
    ap.add_argument("--matrix", default=PATHS['tables']['feature_matrix'],
                    help="Compiled feature matrix parquet.")
    ap.add_argument("--model-dir", default=PATHS['models']['models_dir'],
                    help="Directory with trained fold models.")
    ap.add_argument("--out-dir", default=PATHS['analysis']['attribution_dir'],
                    help="Directory for the analysis outputs.")
    ap.add_argument("--n-clusters", type=int, default=14, help="KMeans clusters (default 14).")
    ap.add_argument("--failure-threshold", type=float, default=0.2,
                    help="grely_prob below which a prediction is a failure case (default 0.2).")
    ap.add_argument("--config", default=None, help="Path to grely.yaml.")
    ap.add_argument("--no-plot", action="store_true", help="Skip the UMAP scatter PNGs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = gdata.load_config(args.config)

    df = pd.read_parquet(args.matrix)
    df, features, _ = gdata.prepare(df, cfg)
    print(f"Loaded matrix: {df.shape[0]} tuples, {len(features)} features.")

    # 1. Out-of-fold SHAP attributions.
    df_test, shap_df = gattr.shap_out_of_fold(df, features, model_dir=args.model_dir, config=cfg)
    print(f"SHAP computed for {len(df_test)} held-out predictions.")

    # 2. Global + family importance.
    gattr.global_importance(shap_df).rename("mean_abs_shap").to_csv(
        os.path.join(args.out_dir, "global_importance.csv"))
    gattr.family_importance(shap_df).rename("mean_abs_shap").to_csv(
        os.path.join(args.out_dir, "family_importance.csv"))

    # 3. Embed + cluster.
    embedding = gattr.umap_embedding(shap_df)
    labels = gattr.cluster_embedding(embedding, n_clusters=args.n_clusters)

    id_cols = [c for c in ("variant", "gene_id", "target_tissue", "fold",
                           "grely_prob", "logSum_sign_correct") if c in df_test.columns]
    emb = df_test[id_cols].copy()
    emb["umap1"] = embedding[:, 0]
    emb["umap2"] = embedding[:, 1]
    emb["cluster"] = labels
    emb.to_csv(os.path.join(args.out_dir, "embedding.csv"), index=False)

    # 4. Per-cluster summary + failure-mode enrichment.
    summary = gattr.cluster_summary(df_test, shap_df, labels)
    summary.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False)

    enrichment = gattr.failure_enrichment(df_test, shap_df, threshold=args.failure_threshold)
    enrichment.to_csv(os.path.join(args.out_dir, "failure_enrichment.csv"), index=False)

    # 5. Optional plots.
    if not args.no_plot:
        gattr.plot_embedding(embedding, color=emb["grely_prob"].values,
                             out_path=os.path.join(args.out_dir, "embedding_by_grely_score.png"),
                             title="UMAP of SHAP attributions — gRely score")
        gattr.plot_embedding(embedding, color=labels, categorical=True,
                             out_path=os.path.join(args.out_dir, "embedding_by_cluster.png"),
                             title="UMAP of SHAP attributions — clusters")
        if "logSum_sign_correct" in emb.columns:
            gattr.plot_embedding(embedding, color=emb["logSum_sign_correct"].astype(int).values,
                                 categorical=True,
                                 out_path=os.path.join(args.out_dir, "embedding_by_correctness.png"),
                                 title="UMAP of SHAP attributions — sign concordance")

    # Console summary.
    print(f"\nCluster mean gRely score ranges "
          f"{summary['mean_grely_score'].min():.3f}–{summary['mean_grely_score'].max():.3f} "
          f"across {len(summary)} clusters.")
    print("Top failure-enriched feature families:")
    for _, r in enrichment.head(5).iterrows():
        print(f"  {r['family']:<34s} enrichment={r['enrichment']:.2f}")
    print(f"\nWrote analysis outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
