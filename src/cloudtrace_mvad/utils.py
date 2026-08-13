from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def stable_hash(value: Any, length: int = 16) -> str:
    text = "" if value is None else str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(obj: Any, path: str | Path) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=convert)
