# ===========================================================================
# gRely — tool workflow
#
# Stages: (1) extract the feature matrix, (2) train gRely, (3) apply it to other
# S2F models, (4) analyze its attributions, (5) evaluate its reliability. The
# feature-extraction stages are SLURM ARRAY jobs over tens of thousands of
# (variant / gene) shards and depend on cluster-resident reference data (hg38,
# Borzoi weights, TFRecords, eQTL Catalogue); the ready-made array scripts are
# under scripts/slurm/. Training and everything downstream run on a single
# machine from the compiled matrix.
#
#   [01 labels, 02 preds, 03 perf] -> 04 featurize   -> MATRIX
#   MATRIX -> 05 train_grely                          -> models/{fold}.json
#   MATRIX + other-model VEPs -> 06 apply             -> transfer report
#   MATRIX + models -> 07 attribution                 -> SHAP / UMAP / clusters
#   MATRIX + models -> 08 evaluate                    -> coverage / accuracy / per-tissue
#
# Config: configs/paths.yaml (data roots), configs/grely.yaml (model / folds).
# ===========================================================================

PY ?= python
MATRIX ?= data/task_sign_matching_with_features.parquet
MODELS ?= models
SHARD ?= 0

.PHONY: help env labels predictions performance features \
        train apply-alphagenome apply-enformer alphagenome-vep enformer-vep \
        attribution evaluate clean-pyc

help:
	@echo "gRely stages:"
	@echo "  make env                 # editable install of the grely package"
	@echo "  make labels              # 01: curate positive/negative eQTLs (SLURM)"
	@echo "  make predictions         # 02: Borzoi ref/alt predictions (GPU, SLURM)"
	@echo "  make performance         # 03: Borzoi vs observed Pearson (SLURM)"
	@echo "  make features            # 04: compile the ~1,161-feature matrix (SLURM)"
	@echo "  make train               # 05: train gRely on \$$MATRIX -> \$$MODELS/"
	@echo "  make alphagenome-vep     # 06: compute AlphaGenome VEPs"
	@echo "  make enformer-vep        # 06: compute Enformer VEPs"
	@echo "  make apply-alphagenome   # 06: apply gRely to AlphaGenome VEPs (transfer report)"
	@echo "  make apply-enformer      # 06: apply gRely to Enformer VEPs (transfer report)"
	@echo "  make attribution         # 07: SHAP / UMAP / clusters / failure modes"
	@echo "  make evaluate            # 08: coverage / accuracy / Table 1 / per-tissue"
	@echo "Vars: MATRIX=$(MATRIX)  MODELS=$(MODELS)  SHARD=$(SHARD)"

env:
	$(PY) -m pip install -e .

# --- Stage 01-04: feature extraction (SLURM arrays; see scripts/slurm/) -------
labels:
	@echo ">> 01 labels:  sbatch scripts/slurm/find_pos.sh ; sbatch scripts/slurm/find_neg.sh"
	@echo "   single shard: cd scripts/01_labels && $(PY) find_pos.py $(SHARD)"

predictions:
	@echo ">> 02 Borzoi predictions (GPU): sbatch scripts/slurm/borzoi_predictions_usual.sh"
	@echo "   + borzoi_predictions_difference_only_per_track_usual.sh, borzoi_predictions_genes_usual.sh"

performance:
	@echo ">> 03 Borzoi performance (GPU): sbatch scripts/slurm/borzoi_performance_usual.sh"
	@echo "   + borzoi_performance_genes_usual.sh"

features:
	@echo ">> 04 featurization: sbatch scripts/slurm/variant_features_usual.sh ;"
	@echo "   sbatch scripts/slurm/gene_data_features_braid.sh ; sbatch scripts/slurm/compile_final_features_usual.sh"
	@echo "   single shard: cd scripts/04_featurization && $(PY) compile_final_feature_set.py $(SHARD)"
	@echo "   then concatenate final_feature_set/*.parquet + labels -> $(MATRIX)"

# --- Stage 05: train gRely ---------------------------------------------------
train:
	$(PY) scripts/05_train_grely/train_grely.py --matrix $(MATRIX) --out-models $(MODELS) --out-prob yprob

# --- Stage 06: apply to other models -----------------------------------------
alphagenome-vep:
	@echo ">> edit paths inside scripts/06_apply_other_models/alphagenome_vep.py, then:"
	@echo "   cd scripts/06_apply_other_models && $(PY) alphagenome_vep.py"

enformer-vep:
	$(PY) scripts/06_apply_other_models/enformer_vep.py \
	  --matrix $(MATRIX) --pred-dir data/predictions --gtf data/gencode41_basic_nort.gtf \
	  --out-csv data/enformer_pred.csv

apply-alphagenome:
	$(PY) scripts/06_apply_other_models/apply_grely.py --matrix $(MATRIX) \
	  --other-model-csv data/alphagenome_pred.csv --sign-col alphagenome_sign_concordant \
	  --model-dir $(MODELS)

apply-enformer:
	$(PY) scripts/06_apply_other_models/apply_grely.py --matrix $(MATRIX) \
	  --other-model-csv data/enformer_pred.csv --sign-col enformer_sign_concordant \
	  --model-dir $(MODELS)

# --- Stage 07: analyze feature attributions (SHAP / UMAP / clusters) ---------
attribution:
	$(PY) scripts/07_analyze_attributions/analyze_attributions.py \
	  --matrix $(MATRIX) --model-dir $(MODELS)

# --- Stage 08: evaluate reliability (coverage / accuracy / per-tissue) --------
evaluate:
	$(PY) scripts/08_evaluate/evaluate_grely.py \
	  --matrix $(MATRIX) --model-dir $(MODELS)

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	find . -name '*.py[cod]' -delete
