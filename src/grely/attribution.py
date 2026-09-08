"""Feature-attribution analysis for gRely.

Once gRely is trained, its per-prediction SHAP attributions explain *why* a
prediction was judged reliable or not. This module turns those attributions into
an analysis workflow:

1. **Out-of-fold SHAP** — attribute each held-out prediction with the fold model
   that never saw it (`shap_out_of_fold`), giving an (n_predictions x n_features)
   matrix of log-odds contributions.
2. **Embedding** — embed predictions in 2-D from their SHAP vectors with UMAP
   (`umap_embedding`) and cluster them (`cluster_embedding`).
3. **Ranking** — global feature importance (`global_importance`) and importance
   aggregated into feature families (`family_importance`).
4. **Failure modes** — per-cluster summaries (`cluster_summary`) and, for
   low-confidence predictions, which feature families are over-represented
   (`failure_enrichment`).

Heavy, optional dependencies (`shap`, `umap-learn`, `scikit-learn`,
`matplotlib`) are imported lazily inside the functions that need them, so the
rest of the package imports without them. Install with the `interpret` extra:
``pip install -e ".[interpret]"``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import apply as _apply
from . import data as _data

# Default random seed, shared with the rest of the project.
SEED = 42


# ---------------------------------------------------------------------------
# SHAP attribution
# ---------------------------------------------------------------------------

def compute_shap(model, X) -> np.ndarray:
    """Per-prediction SHAP values (log-odds) for a trained gRely XGBoost model.

    Returns an (n_samples, n_features) array of contributions toward
    ``P(prediction correct)`` on the log-odds scale.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):          # some shap versions return [neg, pos]
        sv = sv[1]
    return np.asarray(sv)


def shap_out_of_fold(df: pd.DataFrame, features: list[str], model_dir: str = "models",
                     config: dict | None = None):
    """Attribute every held-out prediction with the model that held it out.

    For each fold, the model trained without that fold's tissues + test
    chromosomes attributes exactly that fold's test tuples, so no attribution
    comes from data the model saw during training. Returns
    ``(df_test, shap_df)``:

    - ``df_test``: the pooled test tuples, with added ``grely_prob`` and ``fold``.
    - ``shap_df``: SHAP values, columns == ``features``, index aligned to df_test.
    """
    cfg = config or _data.load_config()
    folds = _data.get_folds(cfg)
    train_chroms, test_chroms = _data.get_chroms(cfg)

    parts_df, parts_shap = [], []
    for fold_name, test_tissues in folds:
        train_tissues = _data.union(*[s for (n, s) in folds if n != fold_name])
        _, test_mask = _data.split_masks(df, test_tissues, train_tissues, train_chroms, test_chroms)
        if not test_mask.any():
            continue
        model_path = os.path.join(model_dir, f"xgb_model_center_196kb_{fold_name}_chromosomal.json")
        model = _apply.load_model(model_path)

        X_test = df.loc[test_mask, features]
        sv = compute_shap(model, X_test)

        d = df.loc[test_mask].copy()
        d["grely_prob"] = model.predict_proba(X_test)[:, 1]
        d["fold"] = fold_name
        parts_df.append(d)
        parts_shap.append(pd.DataFrame(sv, columns=features, index=d.index))

    if not parts_df:
        raise RuntimeError("No test tuples found for any fold — check the matrix and config.")
    df_test = pd.concat(parts_df)
    shap_df = pd.concat(parts_shap)
    return df_test, shap_df


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------

# Ordered (substring, family) rules — first match wins. The gRely feature names
# are prefixed by the block that produced them (see docs/FEATURES.md), so a
# small set of prefix rules groups the ~1,161 features into interpretable
# families. Adjust to taste; nothing downstream hard-codes the family names.
_FAMILY_RULES = [
    ("model_predictions_difference_all_tracks_", "VEP across all tracks"),
    ("model_predictions_gene_",                  "Model prediction (gene)"),
    ("model_predictions_",                       "VEP magnitude / model uncertainty"),
    ("variant_performance_",                     "Base-model accuracy (variant)"),
    ("gene_performance_",                        "Base-model accuracy (gene)"),
    ("gene_observed_data_",                      "Gene expression (observed)"),
    ("variant_sequence_",                        "Chromatin / sequence at variant"),
    ("gene_sequence_",                           "Chromatin / sequence at gene"),
    ("phyloP",                                   "Conservation"),
    ("num_pos_eqtl_",                            "Local eQTL density"),
    ("avg_logSum",                               "VEP magnitude (target tissue)"),
]


def feature_family(name: str) -> str:
    """Map a feature name to a coarse family (heuristic, prefix-based)."""
    for token, family in _FAMILY_RULES:
        if token in name:
            return family
    return "Variant / genomic context"


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature, most important first."""
    return shap_df.abs().mean(axis=0).sort_values(ascending=False)


def family_importance(shap_df: pd.DataFrame) -> pd.Series:
    """Total mean |SHAP| per feature family, most important first."""
    fam = pd.Series({f: feature_family(f) for f in shap_df.columns})
    return shap_df.abs().mean(axis=0).groupby(fam).sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Embedding + clustering
# ---------------------------------------------------------------------------

def umap_embedding(shap_df: pd.DataFrame, n_neighbors: int = 15, min_dist: float = 0.1,
                   metric: str = "euclidean", random_state: int = SEED) -> np.ndarray:
    """2-D UMAP embedding of predictions from their SHAP attribution vectors."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=random_state)
    return reducer.fit_transform(shap_df.values)


def cluster_embedding(embedding: np.ndarray, n_clusters: int = 14,
                      random_state: int = SEED) -> np.ndarray:
    """KMeans cluster labels for a 2-D embedding."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(embedding)


# ---------------------------------------------------------------------------
# Cluster / failure-mode analysis
# ---------------------------------------------------------------------------

def cluster_summary(df_test: pd.DataFrame, shap_df: pd.DataFrame, labels,
                    score_col: str = "grely_prob", sign_col: str = "logSum_sign_correct",
                    top_n: int = 10) -> pd.DataFrame:
    """One row per cluster: size, mean gRely score, sign concordance, top features.

    ``top_features`` lists the ``top_n`` features with the largest mean |SHAP|
    within the cluster — i.e. what that region of the embedding is driven by.
    """
    labels = np.asarray(labels)
    rows = []
    for c in sorted(np.unique(labels)):
        mask = labels == c
        sub_shap = shap_df.loc[df_test.index[mask]]
        top = sub_shap.abs().mean(axis=0).sort_values(ascending=False).head(top_n)
        row = {
            "cluster": int(c),
            "n": int(mask.sum()),
            "mean_grely_score": float(df_test.loc[mask, score_col].mean()),
            "top_features": "; ".join(top.index),
        }
        if sign_col in df_test.columns:
            row["sign_concordance"] = float(df_test.loc[mask, sign_col].astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_grely_score", ascending=False).reset_index(drop=True)


def failure_enrichment(df_test: pd.DataFrame, shap_df: pd.DataFrame,
                       score_col: str = "grely_prob", threshold: float = 0.2) -> pd.DataFrame:
    """Which feature families are over-represented among low-confidence predictions.

    Failure cases are predictions with ``grely_prob < threshold``. For each
    family, the share of total mean |SHAP| importance among failures is divided
    by its share across all predictions; ``enrichment > 1`` means the family is
    disproportionately informative for low-confidence calls.
    """
    fam = pd.Series({f: feature_family(f) for f in shap_df.columns})

    def shares(idx):
        imp = shap_df.loc[idx].abs().mean(axis=0).groupby(fam).sum()
        total = imp.sum()
        return imp / total if total > 0 else imp

    overall = shares(df_test.index)
    fail_idx = df_test.index[df_test[score_col] < threshold]
    if len(fail_idx) == 0:
        raise RuntimeError(f"No predictions with {score_col} < {threshold}.")
    failure = shares(fail_idx)

    out = pd.DataFrame({"overall_frac": overall, "failure_frac": failure}).fillna(0.0)
    out["enrichment"] = out["failure_frac"] / out["overall_frac"].replace(0.0, np.nan)
    out.index.name = "family"
    return out.sort_values("enrichment", ascending=False).reset_index()


# ---------------------------------------------------------------------------
# Optional: a plain embedding scatter (utility, not a publication figure)
# ---------------------------------------------------------------------------

def plot_embedding(embedding: np.ndarray, color=None, out_path: str | None = None,
                   title: str | None = None, cmap: str = "viridis", categorical: bool = False):
    """Save a simple 2-D scatter of the embedding, colored by ``color``.

    A quick look at the embedding — colour by gRely score, sign concordance, or
    cluster label. Returns the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if color is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=4, alpha=0.6)
    elif categorical:
        color = np.asarray(color)
        for c in np.unique(color):
            m = color == c
            ax.scatter(embedding[m, 0], embedding[m, 1], s=4, alpha=0.6, label=str(c))
        ax.legend(markerscale=3, fontsize=7, loc="best", frameon=False)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=color, s=4, alpha=0.7, cmap=cmap)
        fig.colorbar(sc, ax=ax)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    return fig
