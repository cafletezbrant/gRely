# gRely restructuring — MIGRATION log

How the original, unorganized code (three sibling folders plus `grely_paper.pdf`)
was reorganized into this repository.

> **Goal:** a reusable **tool** repo focused on (1) feature extraction, (2)
> training gRely, and (3) applying it to other S2F models. Reproducing the
> paper's specific figures is explicitly **out of scope** — figure-generation
> notebooks/code were removed, and the non-figure logic they contained
> (training, cross-model application) was extracted into the library + CLIs.

> **Scope constraint honored:** only this `gRely_github/` folder was created;
> the original folders were read but never modified. Every feature-extraction
> file here is byte-identical to an original (md5-verified); other modules are
> faithful extractions (see Verification).

> **No reference repo was provided**, so the layout is the standard research/tool
> layout (`src/ scripts/ configs/ docs/`).

---

## Phase 0 — Inventory (source → destination)

| Original file | New location / disposition |
|---|---|
| `gRely_borzoi_and_complete_featurization_codes/find_pos.py`, `find_neg.py` | `scripts/01_labels/` (verbatim) |
| `…/borzoi_predictions*.py` (4) | `scripts/02_borzoi_predictions/` (verbatim) |
| `…/borzoi_performance*.py` (2) | `scripts/03_borzoi_performance/` (verbatim) |
| `…/variant_features.py`, `gene_data_features.py`, `compile_final_feature_set.py` | `scripts/04_featurization/` (verbatim) |
| `…/*.sh` (17) | `scripts/slurm/` (verbatim) |
| `applying_gRely_to_other_models/alphagenome.py` | `scripts/06_apply_other_models/alphagenome_vep.py` (verbatim) |
| `applying_gRely_to_other_models/enformer_plots.ipynb` | **removed** (figures); VEP logic → `enformer_vep.py` + `grely.other_models` |
| `final_notebooks_…/figure_1.ipynb` | **removed** (figures); training logic → `grely.data` / `grely.train` / `train_grely.py` |
| `final_notebooks_…/figure_2_gRely.ipynb` | **removed** (figures + SHAP; out of scope) |
| `grely_paper.pdf` | `paper/` (reference) |

New code written for the tool:
`src/grely/{__init__,paths,stats,data,train,apply,other_models}.py`,
`scripts/05_train_grely/train_grely.py`,
`scripts/06_apply_other_models/{apply_grely,enformer_vep}.py`,
`configs/{paths,grely}.yaml`, `docs/`, packaging + Makefile.

---

## Phase 2 — Classification

| Label | Files |
|---|---|
| `reproduces` (paper figures) | the three notebooks — **removed** (out of scope); training/apply logic extracted first |
| `pipeline` (produces an intermediate) | `scripts/01_labels` … `04_featurization`; `scripts/06_apply_other_models/*_vep.py` |
| `shared` (utility used by ≥2) | helpers duplicated across `gene_data_features.py` & `compile_final_feature_set.py` → `src/grely/stats.py` (verbatim); gene-slice geometry → `src/grely/other_models.py` (verbatim) |
| `tool` (extracted from notebooks) | `grely.data`, `grely.train`, `grely.apply`, `train_grely.py`, `apply_grely.py` |
| `duplicate` | SLURM `*.sh` cluster variants (kept intentionally) |

### Duplicate helpers & the one divergence (do **not** silently merge)

Functions in **both** `gene_data_features.py` and `compile_final_feature_set.py`:

| Function | Identical? | Action |
|---|---|---|
| `compute_tau`, `compute_gini`, `compute_specificity_metrics` | ✅ identical | verbatim in `src/grely/stats.py` |
| `extract_target_features` | ✅ identical | left inline (couples to the diverging one) |
| `extract_celltype_features` | ⚠️ **DIVERGES** | left inline in **both**, unchanged |

> Divergence: for `celltype_cv_all/exon`, `gene_data_features.py` uses
> `scipy.stats.variation(cell_means)`, whereas `compile_final_feature_set.py`
> uses a stabilized `nanstd/(nanmean+1e-8)`. Kept separate per the
> "don't silently change results" rule. Confirm intent before unifying.

### SLURM script variants (`duplicate`, kept)

Each stage has up to three near-identical submission scripts differing only in
cluster headers: `*.sh` (braid-style GPU), `*_braid.sh` (`owner_gred_braid_gpu`),
`*_usual.sh` (`batch_gpu`/`batch_cpu`, `--qos=1d`). All kept under
`scripts/slurm/`; the `*_usual.sh` variants are canonical in the Makefile.

---

## Phase 5 — Hardcoded paths → `configs/paths.yaml`

Values copied verbatim (scripts **not** edited). Distinct roots:

| Root | Meaning |
|---|---|
| `/home/rafia1/scratch/data` (≡ `../data` from `$SCRATCH/codes`) | primary working data dir |
| `/home/rafia1/scratch/borzoi/examples` | Borzoi checkout (`borzoi_helpers`, `params_pred.json`, targets) |
| `/gstore/data/humgenet/ref/eqtl_catalogue` | eQTL Catalogue SuSiE + sumstats (read-only) |
| `/gstore/data/resbioai/grelu/borzoi-data/hg38` | Borzoi `sequences.bed` + TFRecords (read-only) |

The extracted tool modules read paths/config from `configs/*.yaml`
(`grely.paths.load_paths`, `grely.data.load_config`). The verbatim
feature-extraction scripts still contain their original hardcoded paths — wiring
them to the config is a validated next step, not done blind.

---

## Phase 8 — Verification status  ⚠️ READ THIS

**No numerical equivalence could be verified in this environment** — no GPU, no
Borzoi weights, no hg38/TFRecords, no eQTL Catalogue access, and no compiled
feature matrix, so no stage could be executed end to end.

What *was* verified:
- **Feature-extraction scripts + SLURM scripts + AlphaGenome VEP script are
  byte-identical to their originals** (md5).
- The 8 helpers in `src/grely/stats.py` and the 4 gene-slice functions in
  `src/grely/other_models.py` are **AST-verbatim** copies of the originals.
- All Python modules/scripts **compile** (`py_compile`); the `grely` package
  **imports without xgboost** (train/apply import it lazily); `configs/*.yaml`
  **parse and resolve**; and the non-model code paths were exercised on a
  synthetic feature frame — feature selection (leakage columns correctly
  dropped), the center-window filter, disjoint joint tissue+chromosome fold
  masks, and `transfer_eval`. XGBoost was **not installed** in the restructuring
  env, so the actual `fit`/`save_model`/`predict_proba` calls in `grely.train` /
  `grely.apply` were logic-reviewed but not executed here.

Faithful-but-not-byte-identical (extracted from the removed notebooks):
`grely.data`, `grely.train`, `grely.apply`, and the two CLIs. These preserve the
computation — same feature selection (`df.columns[40:] + avg_logSum` minus the
10 leakage columns), same center-window filter (`<196608//2`), same joint
tissue+chromosome folds, same XGBoost hyperparameters/seed, same per-fold
`scale_pos_weight`, and the same logSUM VEP formulas — but are new code.

Unverified / next steps (require a live environment + the feature matrix):
1. Run `train_grely.py` on the real matrix and confirm per-fold models match a
   notebook-trained baseline (identical hyperparameters + seed ⇒ identical trees).
2. Wire the stage-01–04 scripts to `configs/paths.yaml` and `grely.stats`, then
   re-run and diff outputs — do not do this blind.
3. Resolve the missing gene-sequence-feature producer and the pkl→parquet step
   (below).

---

## Known gaps carried over from the original code

- **Orphan input — missing producer.** `compile_final_feature_set.py` reads
  `gene_sequence_features_new/{gene}.parquet` (gene *sequence* features), but no
  supplied script produces it (`variant_features.py` produces *variant* sequence
  features only). Confirm with the author; **do not fabricate**.
- **Format/naming mismatch.** `variant_features.py` writes
  `sequence_features/{variant}.pkl`; `compile` reads
  `sequence_features_new/{variant}.parquet` — a pkl→parquet convert+rename step
  is implied but not present. Both names are documented in `configs/paths.yaml`.
- **Master-matrix assembly** (`task_sign_matching_with_features.parquet`) — the
  concatenation of `final_feature_set/*.parquet` with labels happened in a
  now-removed notebook cell; it is a short glue step, not a supplied script.

## Informational paper notes (not code issues)

- Abstract/Methods say **1,161** features; Fig 1C legend says **1121**.
- The Enformer Methods paragraph in the PDF contains an author TODO.
