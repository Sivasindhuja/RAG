"""Load prompts and runtime settings from YAML."""

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent

with open(_CONFIG_DIR / "prompts.yaml", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

with open(_CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)
