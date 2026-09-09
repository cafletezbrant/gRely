# gRely

**Reliability scoring for genome-trained sequence-to-function (S2F) model predictions.**

gRely is a meta-model that assigns a per-prediction confidence score —
`P(base-model VEP is correct) ∈ [0, 1]` — to individual variant effect
predictions (VEPs) from S2F models such as Borzoi. Instead of the usual heuristic
of keeping only large-magnitude predictions (which discards exactly the
low-effect variants where most GWAS signal lives), gRely learns, from **~1,161
multimodal features** describing the variant, gene, tissue, and the base model's
own behavior, whether a given prediction can be trusted. Trained on Borzoi VEPs
labeled by sign concordance against SuSiE fine-mapped GTEx eQTLs, it reaches
**97% sign concordance in its top 20%** of held-out predictions, stays
discriminative among low-magnitude variants, and its Borzoi-trained scores
**transfer zero-shot** to AlphaGenome and Enformer.

This repository packages gRely as a **tool** with five stages:

1. **Feature extraction** — build the (variant, gene, tissue) feature matrix.
2. **Training** — fit the gRely meta-model with the paper's cross-validation.
3. **Applying to other models** — score any S2F model's VEPs, no retraining.
4. **Feature-attribution analysis** — explain gRely with per-prediction SHAP,
   embed and cluster predictions (UMAP), and analyze failure modes.
5. **Evaluation** — reliability vs. coverage, gRely vs. effect-size filtering,
   and per-tissue reliability.

> Reference: Rafi, Eraslan, Fletez-Brant, *gRely*. This repository is the
> reusable implementation of the method.

---

## Repository layout

```
gRely_github/
├── README.md
├── pyproject.toml            # `pip install -e .` installs the `grely` package
├── requirements.txt / environment.yml
├── Makefile                  # extract -> train -> apply -> analyze
├── configs/
│   ├── paths.yaml            # all data / reference / output paths
│   └── grely.yaml            # model hyperparameters, folds, tissue sets, features
├── src/grely/                # the importable library
│   ├── stats.py              # shared numerical helpers
│   ├── paths.py              # config-driven path resolution
│   ├── data.py               # feature/label selection + CV fold masks
│   ├── train.py              # train the XGBoost meta-model per fold
│   ├── apply.py              # score a feature table; cross-model transfer eval
│   ├── other_models.py       # AlphaGenome / Enformer gene-slice geometry
│   ├── attribution.py        # SHAP attributions, UMAP, clustering, failure modes
│   └── evaluate.py           # coverage/accuracy, gRely-vs-magnitude, per-tissue
├── scripts/
│   ├── 01_labels/            # curate positive/negative eQTLs
│   ├── 02_borzoi_predictions/# Borzoi ref/alt + score predictions (GPU)
│   ├── 03_borzoi_performance/# Borzoi vs observed Pearson (GPU)
│   ├── 04_featurization/     # variant/gene features + compile the matrix
│   ├── 05_train_grely/       # train_grely.py  (CLI)
│   ├── 06_apply_other_models/# alphagenome_vep.py, enformer_vep.py, apply_grely.py
│   ├── 07_analyze_attributions/# analyze_attributions.py  (CLI)
│   ├── 08_evaluate/          # evaluate_grely.py  (CLI)
│   └── slurm/                # SLURM array submission scripts
├── docs/PIPELINE.md          # stage-by-stage inputs/outputs
├── docs/FEATURES.md          # the feature families
└── data/                     # gitignored; see data/README.md
```

---

## Install

```bash
conda env create -f environment.yml     # or: python -m venv .venv && pip install -e .
conda activate grely
pip install -e .                         # installs the `grely` package
```

The **Borzoi prediction stage** additionally needs a GPU, TensorFlow, and the
Borzoi package (`baskerville`, `borzoi_helpers`) from
<https://github.com/calico/borzoi> plus the Borzoi weights. Training and applying
gRely need only the light dependencies above. The **feature-attribution
analysis** (stage 07) needs the `interpret` extra:

```bash
pip install -e ".[interpret]"    # shap, umap-learn, scikit-learn, matplotlib
```

---

## Quickstart

Everything is driven by two config files: `configs/paths.yaml` (all data,
reference, and output locations — every root defaults to a folder under `data/`,
so populate `data/` or repoint a root) and `configs/grely.yaml` (model, folds,
and feature selection).

### 1. Feature extraction (produces the matrix)

The feature pipeline runs as SLURM array jobs (`make labels predictions
performance features`, or `sbatch scripts/slurm/*.sh`) and writes one parquet per
(variant, gene, tissue) tuple under `final_feature_set/`. Concatenating those with
the sign-concordance label gives the master matrix
`task_sign_matching_with_features.parquet`. See [`docs/PIPELINE.md`](docs/PIPELINE.md).

### 2. Train gRely

```bash
python scripts/05_train_grely/train_grely.py \
    --matrix data/task_sign_matching_with_features.parquet \
    --out-models models --out-prob yprob
# or: make train MATRIX=... MODELS=models
```

Trains one XGBoost model per fold with a joint **tissue + chromosome** hold-out
(hold out one of five GTEx tissue sets *and* chr1–3; train on the rest and
chr4–X). Hyperparameters (from `configs/grely.yaml`, identical to the paper):
300 estimators, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`,
`colsample_bytree=0.8`, `eval_metric=logloss`, `random_state=42`,
`scale_pos_weight = n_neg/n_pos` per fold.

### 3. Analyze feature attributions

Explain what gRely learned: per-prediction SHAP attributions, a UMAP embedding of
predictions built from those attributions, clusters, and the feature families
that drive low-confidence predictions.

```bash
pip install -e ".[interpret]"
python scripts/07_analyze_attributions/analyze_attributions.py \
    --matrix data/task_sign_matching_with_features.parquet \
    --model-dir models --out-dir analysis/attribution --n-clusters 14
# or: make attribution
```

Attributions are computed **out-of-fold** — each held-out prediction is
attributed by the model that never saw it. Outputs (`analysis/attribution/`):
`global_importance.csv`, `family_importance.csv`, `embedding.csv` (UMAP + cluster
+ gRely score + ids), `cluster_summary.csv` (per-cluster size, mean score, sign
concordance, top features), `failure_enrichment.csv`, and UMAP scatter PNGs.

Programmatic use:

```python
import pandas as pd
from grely import data, attribution as attr

df = pd.read_parquet("data/task_sign_matching_with_features.parquet")
df, features, _ = data.prepare(df)
df_test, shap_df = attr.shap_out_of_fold(df, features, model_dir="models")
emb = attr.umap_embedding(shap_df)
labels = attr.cluster_embedding(emb, n_clusters=14)
print(attr.cluster_summary(df_test, shap_df, labels))     # per-cluster failure modes
print(attr.failure_enrichment(df_test, shap_df))          # families over-represented in failures
```

### 4. Evaluate reliability

Measure how well gRely separates correct from incorrect predictions, and how it
compares to filtering by predicted effect size. Predictions are scored
**out-of-fold** and pooled across the five folds.

```bash
python scripts/08_evaluate/evaluate_grely.py \
    --matrix data/task_sign_matching_with_features.parquet \
    --model-dir models --out-dir analysis/evaluation
# or: make evaluate
```

Outputs (`analysis/evaluation/`): `coverage_accuracy_grely.csv` /
`coverage_accuracy_magnitude.csv` (accuracy + gene coverage vs. retained
fraction, for gRely vs. the |avg_logSum| baseline), `average_precision_by_fold.csv`,
`threshold_table.csv` (effect-size thresholds vs. gRely quantiles, including the
low-effect regime), `per_tissue_accuracy.csv`, `effect_size_correlation.csv`, and
`mean_grely_per_tissue.csv`.

```python
import pandas as pd
from grely import data, apply, evaluate

df = pd.read_parquet("data/task_sign_matching_with_features.parquet")
df, features, _ = data.prepare(df)
scored = apply.predict_out_of_fold(df, features, model_dir="models")   # adds grely_prob + fold
print(evaluate.threshold_table(scored))          # gRely vs. |VEP| filtering (Table 1)
print(evaluate.coverage_curve(scored))           # accuracy + gene coverage vs. coverage
```

---

## Data & reference resources

All inputs are public (paper, Data availability):

| Resource | Source |
|---|---|
| Borzoi weights + processed data | <https://github.com/calico/borzoi>, `gs://borzoi-paper/data` |
| Fine-mapped eQTLs (SuSiE credible sets) | eQTL Catalogue <https://www.ebi.ac.uk/eqtl> |
| GTEx v8 summary statistics | <https://gtexportal.org> |
| Gene annotations | GENCODE v41 <https://www.gencodegenes.org> |
| Chromatin signal tracks | EpiMap <https://compbio.mit.edu/epimap/> (`scripts/slurm/download_files.sh`) |
See [`data/README.md`](data/README.md) for the expected on-disk layout.

---

## Citation

If you use gRely, please cite:

> Rafi AM, Eraslan G, Fletez-Brant K. *gRely: Relyability for genome trained
> sequence-to-function model predictions.* bioRxiv (2026).
> doi:[10.64898/2026.05.23.727431](https://www.biorxiv.org/content/10.64898/2026.05.23.727431v1)

## License

MIT — see [`LICENSE`](LICENSE).
