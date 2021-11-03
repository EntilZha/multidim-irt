"""Copyright (c) Facebook, Inc. and its affiliates."""
import os
from pathlib import Path

import toml

ROOT = Path(os.environ.get("MULTIDIM_ROOT", "./"))

with open(ROOT / "config.toml") as f:
    conf = toml.load(f)
    DATA_ROOT = ROOT / conf['data_dir']