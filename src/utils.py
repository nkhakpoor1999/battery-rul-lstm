from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

def set_seed(seed: int = 42) -> None:
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def now_tag() -> str:
    # safe folder name
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")
