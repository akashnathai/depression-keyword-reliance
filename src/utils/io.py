"""Shared paths and loaders."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
MASKED = ROOT / "data/masked"
PREDS = ROOT / "results/preds"
RUNS = ROOT / "results/runs"
N_CLASSES = {"depsign": 3, "dsl3": 3, "dsl3m": 3, "rdc": 2}


def load_split(mapping: str, split: str, condition: str = "orig") -> pd.DataFrame:
    base = PROC if condition == "orig" else MASKED / condition
    return pd.read_parquet(base / f"{mapping}_{split}.parquet")


def load_yaml(path) -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_id(model: str, train_set: str, train_cond: str, seed: int) -> str:
    return f"{model}__{train_set}__{train_cond}__s{seed}"


def pred_path(rid: str, test_set: str, test_cond: str) -> Path:
    return PREDS / rid / f"{test_set}__{test_cond}.parquet"
