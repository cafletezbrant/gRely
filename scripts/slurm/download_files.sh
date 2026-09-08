#!/bin/bash
#SBATCH --job-name=dl               # Job name
#SBATCH --array=0-18                       # Array of tasks
#SBATCH --job-name=epimap_imputed_other
#SBATCH --account=batch_cpu    # account name
#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=1                         # Request 8 cores
#SBATCH --mem=16G                          # Memory total in MB
#SBATCH --time=0-1:00                     # Runtime in D-HH:MM
#SBATCH --output=logs/dl%A_%a.out
#SBATCH --error=logs/dl%A_%a.err


set -euo pipefail

BASE="https://epigenome.wustl.edu/epimap/data/averagetracks_pergroup/"
OUTDIR="data/bigwig_tracks"
mkdir -p "$OUTDIR" logs

# Files to download (indices 0..17)
mapfile -t FILES <<'EOF'
average_ATAC-seq_imputed_Other.bigWig
average_CTCF_imputed_Other.bigWig
average_DNase-seq_imputed_Other.bigWig
average_EP300_imputed_Other.bigWig
average_H2AFZ_imputed_Other.bigWig
average_H3K4me1_imputed_Other.bigWig
average_H3K4me2_imputed_Other.bigWig
average_H3K4me3_imputed_Other.bigWig
average_H3K9ac_imputed_Other.bigWig
average_H3K9me3_imputed_Other.bigWig
average_H3K27ac_imputed_Other.bigWig
average_H3K27me3_imputed_Other.bigWig
average_H3K36me3_imputed_Other.bigWig
average_H3K79me2_imputed_Other.bigWig
average_H4K20me1_imputed_Other.bigWig
average_POLR2A_imputed_Other.bigWig
average_RAD21_imputed_Other.bigWig
average_SMC3_imputed_Other.bigWig
EOF

IDX="${SLURM_ARRAY_TASK_ID}"
FILE="${FILES[$IDX]}"
URL="${BASE}${FILE}"
DEST="${OUTDIR}/${FILE}"
TMP="${DEST}.part"

echo "[$(date)] Task ${IDX}: ${URL}"

# Resume-able download; write to .part then move atomically
wget --tries=10 --continue -O "${TMP}" "${URL}"
mv -f "${TMP}" "${DEST}"

echo "[$(date)] Saved → ${DEST}"
