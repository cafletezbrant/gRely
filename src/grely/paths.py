"""Config-driven path resolution for gRely.

This is **additive** helper code — it does not change any existing pipeline
script. The original scripts hardcode absolute paths (e.g.
``/home/rafia1/scratch/data/...``) and cluster-specific reference stores
(``/gstore/...``). To make the repo runnable elsewhere, this module resolves
every location from ``configs/paths.yaml`` (values there are copied verbatim
from the scripts, so the defaults reproduce the original behavior) with an
optional ``GRELY_CONFIG`` / ``GRELY_DATA`` environment-variable override.

Usage (recommended pattern when you later wire scripts to the config)::

    from grely.paths import load_paths
    P = load_paths()                 # reads configs/paths.yaml
    df = pd.read_csv(P["curated_variants"], sep="\\t")

See ``MIGRATION.md`` for the full hardcoded-path inventory.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a light, ubiquitous dep
    yaml = None

# Repo root = two levels up from this file (src/grely/paths.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paths.yaml"


def load_paths(config_path: str | os.PathLike | None = None) -> dict:
    """Load the path map from ``configs/paths.yaml``.

    Resolution order:
      1. explicit ``config_path`` argument
      2. ``GRELY_CONFIG`` environment variable
      3. ``configs/paths.yaml`` at the repo root
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read configs/paths.yaml — `pip install pyyaml`."
        )
    cfg = config_path or os.environ.get("GRELY_CONFIG") or DEFAULT_CONFIG
    with open(cfg) as fh:
        data = yaml.safe_load(fh)

    roots = data.get("roots", {})
    # Allow a single environment override of the primary data root.
    if "GRELY_DATA" in os.environ:
        roots["data"] = os.environ["GRELY_DATA"]

    def expand(value):
        # Substitute only the known {root} names; leave runtime placeholders
        # such as {rep}, {variant}, {gene}, {study_id}, {dataset_id} intact.
        if isinstance(value, str):
            for name, root in roots.items():
                value = value.replace("{" + name + "}", str(root))
        return value

    resolved = {"roots": roots}
    for section, entries in data.items():
        if section == "roots":
            continue
        if isinstance(entries, dict):
            resolved[section] = {k: expand(v) for k, v in entries.items()}
        else:
            resolved[section] = expand(entries)
    return resolved
