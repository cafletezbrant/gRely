import json
import pandas as pd
import pyfaidx
from baskerville import seqnn
import pysam
import sys
sys.path.append('/home/rafia1/scratch/borzoi/examples')
from borzoi_helpers import *
import numpy as np
import os

# read in the sequences
df=pd.read_csv('/home/rafia1/scratch/data/all_genes.csv', sep='\t')
job_id = int(sys.argv[1])
df=df.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

params_file = '/home/rafia1/scratch/borzoi/examples/params_pred.json'
with open(params_file) as params_open :
    params = json.load(params_open)
    params_model = params['model']

targets_file = '/home/rafia1/scratch/borzoi/examples/targets_gtex.txt'
targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

fasta_path = '/home/rafia1/scratch/data/hg38.fa'
pyfaidx.Faidx(fasta_path)
fasta_open = pysam.Fastafile(fasta_path)

seq_len = 524288

models = {}
for rep_ix in range(1, 5):
    model_file = "/home/rafia1/scratch/data/model_weights/model0_best_" + str(rep_ix) + ".h5"
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(target_index)
    models[rep_ix] = seqnn_model

for seq_id in range(len(df)):
    gene=df.iloc[seq_id]['gene_id']
    chrom=df.iloc[seq_id]['chrom']
    center_pos=df.iloc[seq_id]['center']
    poses = [center_pos]

    start = center_pos - seq_len // 2
    end = center_pos + seq_len // 2

    sequence_one_hot_wt = process_sequence(fasta_open, chrom, start, end)

    for rep_ix, model in models.items():
        path = f'/home/rafia1/scratch/data/borzoi_gene_predictions/model_{rep_ix}_{gene}.npy'

        if os.path.exists(path):
            try:
                np.load(path, allow_pickle=False)
                continue
            except Exception as e:
                print(f"Corrupt or unreadable file detected for gene {gene}, rep {rep_ix}: {e}")
                pass            

        y_wt = model(sequence_one_hot_wt[None, ...])[:, None, ...].astype("float16")

        np.save(f'/home/rafia1/scratch/data/borzoi_gene_predictions/model_{rep_ix}_{gene}.npy', y_wt)