"""Build results/master_results.csv from every saved prediction file (safe to re-run).

One row per (run, test set, test condition). Run ids: model__trainset__traincond__s<seed>.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import collapse_binary, compute_metrics  # noqa: E402
from src.utils.io import N_CLASSES, PREDS, ROOT, load_yaml  # noqa: E402

MASTER = ROOT / "results/master_results.csv"


def parse_run(rid):
    model, train_set, train_cond, seed = rid.split("__")
    return {"run_id": rid, "model": model, "train_set": train_set, "train_cond": train_cond,
            "seed": int(seed.lstrip("s"))}


def score_file(path, cfg):
    df = pd.read_parquet(path)
    test_set, test_cond = path.stem.split("__")
    pcols = sorted(c for c in df.columns if c.startswith("p") and c[1:].isdigit())
    probs = df[pcols].to_numpy()
    k = N_CLASSES[test_set]
    if k == 2:
        probs = collapse_binary(probs)
    m = compute_metrics(df["label"].to_numpy(), probs, k, n_boot=cfg["bootstrap"]["n"],
                        seed=cfg["bootstrap"]["seed"], n_bins=cfg["ece_bins"])
    return {"test_set": test_set, "test_cond": test_cond, **m}


def aggregate(only_new=True):
    cfg = load_yaml("configs/grid.yaml")
    old = pd.read_csv(MASTER) if (MASTER.exists() and only_new) else pd.DataFrame()
    done = set(zip(old.get("run_id", []), old.get("test_set", []), old.get("test_cond", [])))
    rows = []
    for run_dir in sorted(p for p in PREDS.iterdir() if p.is_dir()):
        info = parse_run(run_dir.name)
        for f in sorted(run_dir.glob("*.parquet")):
            ts, tc = f.stem.split("__")
            if (info["run_id"], ts, tc) in done:
                continue
            rows.append({**info, **score_file(f, cfg)})
    new = pd.DataFrame(rows)
    allr = pd.concat([old, new], ignore_index=True) if len(old) else new
    if len(allr):
        allr["in_domain"] = allr["train_set"] == allr["test_set"]
        allr = allr.sort_values(["model", "train_set", "train_cond", "seed", "test_set", "test_cond"])
        MASTER.parent.mkdir(parents=True, exist_ok=True)
        allr.to_csv(MASTER, index=False, float_format="%.5f")
    return allr, len(rows)


def seed_summary(master: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "train_set", "train_cond", "test_set", "test_cond"]
    g = master.groupby(keys)
    return g.agg(n_seeds=("seed", "nunique"), macro_f1=("macro_f1", "mean"),
                 macro_f1_sd=("macro_f1", "std"), lo=("macro_f1_lo", "mean"),
                 hi=("macro_f1_hi", "mean"), ece=("ece", "mean")).reset_index()


def krg_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Keyword-reliance gap for orig-trained models: F1(mask-random test) - F1(mask-topic test)."""
    s = summary[summary["train_cond"] == "orig"].pivot_table(
        index=["model", "train_set", "test_set"], columns="test_cond", values="macro_f1").reset_index()
    s["drop_topic"] = s["orig"] - s["mask-topic"]
    s["drop_topic_sui"] = s["orig"] - s["mask-topic+sui"]
    s["drop_random"] = s["orig"] - s["mask-random"]
    s["KRG"] = s["mask-random"] - s["mask-topic"]
    return s


if __name__ == "__main__":
    full = "--full" in sys.argv
    df, n = aggregate(only_new=not full)
    print(f"scored {n} new prediction files; master has {len(df)} rows -> {MASTER}")
