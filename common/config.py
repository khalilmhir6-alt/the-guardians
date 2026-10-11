"""Load config.json and resolve relative paths against the repo root.

One file (config.json) owns all ports, rates and camera settings.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None):
    cfg_path = Path(path) if path else ROOT / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Resolve every configured path to an absolute path under the repo root.
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = str((ROOT / rel).resolve())
    cfg["paths"]["root"] = str(ROOT)
    return cfg
