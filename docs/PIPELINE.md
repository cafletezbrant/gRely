# gRely pipeline — stage-by-stage I/O

Stages match the `scripts/NN_*` folders. Stages 01–04 (feature extraction) are
SLURM **array** jobs (shard index = `sys.argv[1]`); the `scripts/slurm/*.sh`
wrappers submit them. Stages 05–06 run on a single machine from the compiled
matrix. Paths below use the logical names from `configs/paths.yaml`.

```
 01 labels ─┐
            ├─► 04 featurization ─► task_sign_matching_with_features.parquet ─┬─► 05 train_grely ─► models/{fold}.json
 02 preds ──┤                                                                 └─► 06 apply (AlphaGenome / Enformer)
 03 perf ───┘
```

## 01 — Label generation  (`scripts/01_labels/`)
Curate the sign-concordance training labels from the eQTL Catalogue.

| Script | Reads | Writes | Key filters |
|---|---|---|---|
| `find_pos.py <shard>` | `tables.borzoi_gtex_common`; `eqtl.susie_dir`; `eqtl.sumstats_tpl`; `reference.blacklist` | `outputs.pos_eqtls/{dataset_id}_pos_eqtls.tsv.gz` | SuSiE **PIP > 0.9**, SNVs only, blacklist-excluded, **MAF > 0.01** |
| `find_neg.py <shard>` | same + all GTEx credible sets for the study | `outputs.neg_eqtls/{dataset_id}_neg_eqtls.tsv.gz` | remove all credible variants; SNVs; blacklist; distance-matched (±1 kb) to positives, same genes |

→ merged downstream into `tables.all_curated` (`type ∈ {positive,negative}`).

## 02 — Borzoi predictions  (`scripts/02_borzoi_predictions/`, GPU)
Four Borzoi replicates (`model0_best_{1..4}.h5`), 524,288 bp input centered on the variant.

| Script | Output | Tracks |
|---|---|---|
| `borzoi_predictions.py` / `_2.py` | `outputs.borzoi_predictions/model_{rep}_{ref,alt}__{variant}.npy` | 89 GTEx (`targets_gtex.txt`) |
| `borzoi_predictions_difference_only_per_track.py` | `…/model_{rep}_{sum_score,log_score}_{variant}.npy` | 7611 (`targets_human`) |
| `borzoi_predictions_genes.py` | `outputs.borzoi_gene_predictions/model_{rep}_{gene}.npy` | 89 GTEx |

`_2.py` is the resumable variant (100 variants/shard, skips existing/valid `.npy`);
`borzoi_predictions.py` does 1000/shard without the resume check. The per-track
`log_score`/`sum_score` implement the paper's VEP (Eq. 1).

## 03 — Borzoi performance  (`scripts/03_borzoi_performance/`, GPU)
Pearson correlation of Borzoi predictions vs the observed TFRecord targets, both
**along tracks** (7611) and **along sequence bins** (6144), averaged over reps.

| Script | Reads | Writes |
|---|---|---|
| `borzoi_performance.py` | `tables.all_sequences`; `reference.borzoi_sequences_bed`; `reference.borzoi_tfrecords` | `outputs.borzoi_performance/model_{rep}_{variant}_along_{sequence,track}.npy` |
| `borzoi_performance_genes.py` | `tables.all_genes` + same | `outputs.borzoi_gene_performance/model_{rep}_{gene}_along_{sequence,track}.npy` |

## 04 — Featurization  (`scripts/04_featurization/`)
| Script | Writes | Notes |
|---|---|---|
| `variant_features.py <shard>` | `outputs.sequence_features/{variant}.pkl` | k-mer counts (11/21/41), GC (5 windows), EpiMap chromatin stats (11 tracks × 5 windows) |
| `gene_data_features.py <shard>` | `outputs.gene_data_features/{gene}.pkl` | observed-expression features from TFRecord targets (per-GTEx-track) |
| `compile_final_feature_set.py <shard>` | `outputs.final_feature_set/{variant}_{gene}_{tissue}.parquet` | **the master featurizer** — merges 9 feature blocks into one row per tuple |

All shards are concatenated with the labels into
`tables.feature_matrix` = `task_sign_matching_with_features.parquet`
(the concatenation is a short glue step; the master matrix is the interface into
stages 05–06). See [`FEATURES.md`](FEATURES.md) for the blocks.

> Two known gaps (see `MIGRATION.md`): `compile` reads
> `sequence_features_new/{variant}.parquet` (a pkl→parquet rename of stage-04
> output) and `gene_sequence_features_new/{gene}.parquet` (**no producer script
> supplied**).

## 05 — Train gRely  (`scripts/05_train_grely/`)
| Script | Reads | Writes |
|---|---|---|
| `train_grely.py --matrix … --out-models models --out-prob yprob` | `tables.feature_matrix` | `models/xgb_model_center_196kb_{fold}_chromosomal.json`, `yprob/yprob_{fold}.npy` |

Library: `grely.data` (feature/label selection, `prepare`, fold masks) and
`grely.train` (`make_model`, `train_fold`, `train_all_folds`). Hyperparameters,
folds, and feature selection come from `configs/grely.yaml`.

## 06 — Apply to other models  (`scripts/06_apply_other_models/`)
| Script | Reads | Writes |
|---|---|---|
| `alphagenome_vep.py` | `tables.feature_matrix`, `other_models.alphagenome_predictions`, `reference.gencode_gtf` | `other_models.alphagenome_pred_csv` (AG VEP + sign concordance) |
| `enformer_vep.py --matrix … --pred-dir … --gtf …` | `tables.feature_matrix`, `other_models.enformer_predictions`, gtf | `other_models.enformer_pred_csv` (Enformer VEP + sign concordance) |
| `apply_grely.py --matrix … --other-model-csv … --sign-col …` | matrix + a VEP csv + `models/` | prints overall / top-20% / bottom-20% sign concordance; optional scored csv |

Library: `grely.other_models` (AlphaGenome/Enformer gene-slice geometry) and
`grely.apply` (`load_model`, `score`, `score_by_fold`, `transfer_eval`). The
Borzoi-trained fold models are applied **without retraining** — each fold's model
scores exactly its held-out (chr1–3 ∩ held-out-tissues) tuples.
