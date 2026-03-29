"""Shared I/O helpers for parquet, JSON, pickle, and tarball operations."""

from __future__ import annotations

import json
import pickle
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_frame(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    return path


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_pickle(obj: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)
    return path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def extract_tarball(tar_path: Path, destination_dir: Path) -> None:
    ensure_dir(destination_dir)
    with tarfile.open(tar_path, "r:gz") as tar_handle:
        tar_handle.extractall(destination_dir)
