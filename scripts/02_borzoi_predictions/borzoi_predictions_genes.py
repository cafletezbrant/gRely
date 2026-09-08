import json
import pandas as pd
import pyfaidx
from baskerville import seqnn
import pysam
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from grely.paths import load_paths
P = load_paths()
sys.path.append(P['roots']['borzoi_examples'])
from borzoi_helpers import *
import numpy as np
import os

gene_pred_dir = P['outputs']['borzoi_gene_predictions']
os.makedirs(gene_pred_dir, exist_ok=True)

# read in the sequences
df=pd.read_csv(P['tables']['all_genes'], sep='\t')
job_id = int(sys.argv[1])
df=df.iloc[job_id*100:job_id*100+100].reset_index(drop=True)

params_file = P['borzoi']['params_pred']
with open(params_file) as params_open :
    params = json.load(params_open)
    params_model = params['model']

targets_file = P['borzoi']['targets_gtex']
targets_df = pd.read_csv(targets_file, index_col=0, sep='\t')
target_index = targets_df.index

fasta_path = P['reference']['fasta']
pyfaidx.Faidx(fasta_path)
fasta_open = pysam.Fastafile(fasta_path)

seq_len = 524288

models = {}
for rep_ix in range(1, 5):
    model_file = P['borzoi']['model_weights'].format(rep=rep_ix)
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
        path = os.path.join(gene_pred_dir, f'model_{rep_ix}_{gene}.npy')

        if os.path.exists(path):
            try:
                np.load(path, allow_pickle=False)
                continue
            except Exception as e:
                print(f"Corrupt or unreadable file detected for gene {gene}, rep {rep_ix}: {e}")
                pass            

        y_wt = model(sequence_one_hot_wt[None, ...])[:, None, ...].astype("float16")

        np.save(os.path.join(gene_pred_dir, f'model_{rep_ix}_{gene}.npy'), y_wt)