# data/

This directory is the default data root for the pipeline: `configs/paths.yaml`
`roots.data` resolves here (relative to the repo root). Its contents are
**gitignored** — populate it from the public sources listed in the top-level
README. To store data elsewhere, point a root at an absolute path in
`configs/paths.yaml`, or set the `GRELY_DATA` environment variable.

Expected layout (logical names in parentheses match `configs/paths.yaml`):

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
├── eqtl_catalogue/{susie,sumstats}/       (roots.eqtl_catalogue)  # eQTL Catalogue
├── borzoi_data/{sequences.bed,tfrecords}/ (roots.borzoi_data)     # Borzoi training data
│
├── borzoi_gtex_common.csv          (tables.borzoi_gtex_common)    # study/dataset index
├── all_curated.csv                 (tables.all_curated)           # pos+neg tuples
├── all_sequences.csv               (tables.all_sequences)         # variant list
├── all_genes.csv                   (tables.all_genes)             # gene list
├── borzoi_tracks_matched_gtex.csv  (tables.borzoi_tracks_matched)
├── gene_to_tss.csv                 (tables.gene_to_tss)
├── task_sign_matching_with_features.parquet  (tables.feature_matrix)  # stage 04 -> master matrix
│
├── pos_eqtls/  neg_eqtls/                                         # stage 01
├── borzoi_predictions/  borzoi_gene_predictions/                 # stage 02
├── borzoi_performance/  borzoi_gene_performance/                 # stage 03
├── sequence_features(_new)/  gene_sequence_features_new/         # stage 04
├── gene_data_features_new/                                       # stage 04
└── final_feature_set_latest/       (outputs.final_feature_set)   # stage 04 -> matrix
```

The two reference stores are large external datasets. Place (or symlink) them at
`data/eqtl_catalogue/` and `data/borzoi_data/`, or repoint
`roots.eqtl_catalogue` / `roots.borzoi_data` at wherever you keep them.
