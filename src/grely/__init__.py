"""gRely: reliability scoring for genome-trained sequence-to-function models.

gRely is a meta-model that estimates, for an individual variant effect
prediction (VEP) from an S2F model such as Borzoi, the probability that the
prediction is correct. This package exposes the tool as three stages:

1. **Feature extraction** — `scripts/01_labels` … `scripts/04_featurization`
   build the ~1,161-feature (variant, gene, tissue) matrix. Shared numerical
   helpers live in `grely.stats`.
2. **Training** — `grely.train` fits the XGBoost meta-model with the paper's
   leave-one-tissue-set-and-chr1-3-out cross-validation. Data prep is in
   `grely.data`.
3. **Applying to other models** — `grely.apply` scores any feature table with a
   trained model (including AlphaGenome / Enformer VEPs), and
   `grely.other_models` provides the gene-slice geometry to compute those VEPs.
4. **Feature-attribution analysis** — `grely.attribution` computes per-prediction
   SHAP attributions, embeds and clusters predictions (UMAP + KMeans), and
   analyzes failure modes by feature family.

`grely.evaluate` measures reliability vs. coverage, compares gRely to effect-size
filtering, and reports per-tissue reliability. Path/model configuration is read
from `configs/paths.yaml` (`grely.paths`) and `configs/grely.yaml`
(`grely.data.load_config`).
"""

__version__ = "0.1.0"

from . import stats  # noqa: F401
from . import paths  # noqa: F401
from . import data  # noqa: F401
from . import train  # noqa: F401
from . import apply  # noqa: F401
from . import other_models  # noqa: F401
from . import attribution  # noqa: F401
from . import evaluate  # noqa: F401
