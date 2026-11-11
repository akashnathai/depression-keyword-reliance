"""Phase 3: encoder grid. Resumable: finished runs are skipped, a crashed run resumes from its
last epoch checkpoint.

  python scripts/phase3_encoders.py --timing            # one run: roberta, depsign, orig, seed 13
  python scripts/phase3_encoders.py                     # full grid in configured model order
  python scripts/phase3_encoders.py --models roberta    # subset
"""
import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.aggregate import aggregate  # noqa: E402
from src.models.train_encoder import run_done, train_run  # noqa: E402
from src.utils.io import RUNS, load_yaml, run_id  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def prune_kept(model, ts, tc, seeds):
    """Keep best_model only for the seed with the highest validation macro-F1."""
    logs = {}
    for s in seeds:
        f = RUNS / run_id(model, ts, tc, s) / "run.json"
        if not f.exists():
            return
        logs[s] = json.loads(f.read_text())["val_macro_f1"]
    best = max(logs, key=logs.get)
    for s in seeds:
        d = RUNS / run_id(model, ts, tc, s) / "best_model"
        if s != best and d.exists():
            shutil.rmtree(d)
    (RUNS / f"kept__{model}__{ts}__{tc}.json").write_text(
        json.dumps({"best_seed": best, "val_macro_f1": logs}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    grid, cfg = load_yaml("configs/grid.yaml"), load_yaml("configs/encoders.yaml")
    keep = {tuple(k) for k in cfg["keep_models"]}
    ts_all, tc_all = grid["test_sets"], grid["test_conditions"]

    if args.timing:
        jobs = [("roberta", "depsign", "orig", 13)]
    else:
        models = args.models or cfg["order"]
        jobs = [(m, ts, tc, s) for m in models for ts in grid["train_sets"]
                for tc in grid["train_conditions"] for s in grid["seeds"]]

    total = len(jobs)
    for n, (m, ts, tc, s) in enumerate(jobs, 1):
        rid = run_id(m, ts, tc, s)
        if run_done(rid, ts_all, tc_all):
            print(f"[{n}/{total}] skip {rid}", flush=True)
            continue
        print(f"[{n}/{total}] start {rid}  {time.strftime('%H:%M:%S')}", flush=True)
        try:
            log = train_run(rid, m, cfg, ts, tc, s, ts_all, tc_all, keep=(m, ts, tc) in keep)
        except Exception:
            traceback.print_exc()
            print(f"[{n}/{total}] FAILED {rid}", flush=True)
            continue
        print(f"[{n}/{total}] done {rid}: val F1={log['val_macro_f1']:.4f} best epoch={log['best_epoch']} "
              f"train={log['train_seconds'] / 60:.1f}min eval={log['eval_seconds'] / 60:.1f}min "
              f"peak VRAM={log['peak_vram_gb']}GB", flush=True)
        aggregate()
        if (m, ts, tc) in keep:
            prune_kept(m, ts, tc, grid["seeds"])


if __name__ == "__main__":
    main()
