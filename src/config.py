"""Config loader.

Returns the YAML config file as a plain dict. All pipeline modules import
``load_config`` to read thresholds — no magic numbers in code.
"""
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a nested plain dict.

    Args:
        path: Path to the YAML file, e.g. ``"configs/default.yaml"``.

    Returns:
        Nested dict matching the YAML structure.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    import json
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    cfg = load_config(cfg_path)
    print(json.dumps(cfg, indent=2))
