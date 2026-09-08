# data/

This directory is **gitignored** — no data is committed. It is the local root
that `configs/paths.yaml` `roots.data` points at (originally
`/home/rafia1/scratch/data`, equivalently `../data` when a script is launched
from `$SCRATCH/codes`).

Populate it from the public sources listed in the top-level README. Expected
layout (logical names in parentheses match `configs/paths.yaml`):

```
data/
├── hg38.fa                         (reference.fasta)              # + .fai
├── hg38-blacklist.v2.bed           (reference.blacklist)
├── hg38.phyloP100way.bw            (reference.phylop_bw)
├── gencode41_basic_nort.gtf        (reference.gencode_gtf)        # GENCODE v41
├── sequences_human.bed.gz          (reference.sequences_bed_gz)   # Borzoi folds
├── model_weights/model0_best_{1..4}.h5   (borzoi.model_weights)   # Borzoi reps
├── targets_human.txt.gz            (borzoi.targets_human_gz)      # 7611 tracks
├── bigwig_tracks/*.bigWig          (bigwig_dir)                   # EpiMap "Other"
│
├── borzoi_gtex_common.csv          (tables.borzoi_gtex_common)    # study/dataset index
├── all_curated.csv                 (tables.all_curated)           # pos+neg tuples
├── all_sequences.csv               (tables.all_sequences)         # variant list
├── all_genes.csv                   (tables.all_genes)             # gene list
├── borzoi_tracks_matched_gtex.csv  (tables.borzoi_tracks_matched) # Table S3
├── gene_to_tss.csv                 (tables.gene_to_tss)
│
├── pos_eqtls/  neg_eqtls/                                         # stage 01
├── borzoi_predictions/  borzoi_gene_predictions/                 # stage 02
├── borzoi_performance/  borzoi_gene_performance/                 # stage 03
├── sequence_features(_new)/  gene_sequence_features_new/         # stage 04
├── gene_data_features_new/                                       # stage 04
└── final_feature_set_latest/       (outputs.final_feature_set)   # stage 04 -> matrix
```

Read-only cluster reference stores (not under `data/`; see `configs/paths.yaml`
`roots.eqtl_catalogue` and `roots.borzoi_data`):
- `/gstore/data/humgenet/ref/eqtl_catalogue/{susie,sumstats}` — eQTL Catalogue
- `/gstore/data/resbioai/grelu/borzoi-data/hg38/{sequences.bed,tfrecords}` — Borzoi
