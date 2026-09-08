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

This repository packages gRely as a **tool** with three stages:

1. **Feature extraction** — build the (variant, gene, tissue) feature matrix.
2. **Training** — fit the gRely meta-model with the paper's cross-validation.
3. **Applying to other models** — score any S2F model's VEPs, no retraining.

> Reference: `paper/grely_paper.pdf` (Rafi, Eraslan, Fletez-Brant). Reproducing
> the paper's *figures* is out of scope here; this repo is the reusable tool.

---

## Repository layout

```
gRely_github/
├── README.md
├── pyproject.toml            # `pip install -e .` installs the `grely` package
├── requirements.txt / environment.yml
├── Makefile                  # extract -> train -> apply
├── configs/
│   ├── paths.yaml            # ALL data/reference paths (values verbatim)
│   └── grely.yaml            # model hyperparameters, folds, tissue sets, features
├── src/grely/                # the importable library
│   ├── stats.py              # shared numerical helpers (verbatim)
│   ├── paths.py              # config-driven path resolution
│   ├── data.py               # feature/label selection + CV fold masks
│   ├── train.py              # train the XGBoost meta-model per fold
│   ├── apply.py              # score a feature table; cross-model transfer eval
│   └── other_models.py       # AlphaGenome / Enformer gene-slice geometry
├── scripts/
│   ├── 01_labels/            # curate positive/negative eQTLs
│   ├── 02_borzoi_predictions/# Borzoi ref/alt + score predictions (GPU)
│   ├── 03_borzoi_performance/# Borzoi vs observed Pearson (GPU)
│   ├── 04_featurization/     # compile the ~1,161-feature matrix
│   ├── 05_train_grely/       # train_grely.py  (CLI)
│   ├── 06_apply_other_models/# alphagenome_vep.py, enformer_vep.py, apply_grely.py
│   └── slurm/                # SLURM array submission scripts
├── docs/PIPELINE.md          # stage-by-stage inputs/outputs
├── docs/FEATURES.md          # the feature families
├── paper/grely_paper.pdf     # reference
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
gRely need only the light dependencies above.

> ⚠️ Dependency versions are **unpinned** — the original cluster env was not
> available during restructuring. Pin from your live environment before relying
> on exact reproduction (`conda env export --no-builds`).

---

## Quickstart

Everything is driven by two config files: `configs/paths.yaml` (point `roots:` at
your machine) and `configs/grely.yaml` (model + folds + feature selection, values
identical to the paper).

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

### 3. Apply to another S2F model (zero-shot transfer)

Compute the other model's VEPs, then score them with the trained gRely models:

```bash
# AlphaGenome: edit paths in the script, then run it to produce alphagenome_pred.csv
python scripts/06_apply_other_models/alphagenome_vep.py

# Enformer: produce enformer_pred.csv from saved Enformer predictions
python scripts/06_apply_other_models/enformer_vep.py \
    --matrix data/task_sign_matching_with_features.parquet \
    --pred-dir data/predictions --gtf data/gencode41_basic_nort.gtf

# Apply gRely and print the top-20% vs bottom-20% sign-concordance report
python scripts/06_apply_other_models/apply_grely.py \
    --matrix data/task_sign_matching_with_features.parquet \
    --other-model-csv alphagenome_pred.csv \
    --sign-col alphagenome_sign_concordant --model-dir models
```

Programmatic use:

```python
import pandas as pd
from grely import data, apply

df = pd.read_parquet("data/task_sign_matching_with_features.parquet")
df, features, _ = data.prepare(df)                       # features + center-window filter
scored = apply.score_by_fold(df, features, model_dir="models")   # adds grely_prob
res = apply.transfer_eval(scored, sign_col="alphagenome_sign_concordant")
print(res["overall_acc"], res["top_acc"], res["bot_acc"])
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
| Reference genome | hg38 (`hg38.fa`), hg38 blacklist v2, phyloP100way |

See [`data/README.md`](data/README.md) for the expected on-disk layout.

---

## Provenance & guarantees

This repository was produced by a behavior-preserving restructuring of the
original research code. The feature-extraction scripts are **byte-identical** to
their originals (md5-verified); the shared helpers in `src/grely/stats.py` and the
`other_models` gene-slice geometry are **AST-verbatim**; and the training / apply
modules are faithful extractions of the notebook logic (same features, folds,
hyperparameters, seed, and VEP formulas). Full numerical equivalence of a re-run
was **not** verifiable in the restructuring environment (no GPU/data) — see
[`MIGRATION.md`](MIGRATION.md) before treating any stage as reproduced.
