"""Show grid progress: finished / remaining runs per model and a rough ETA.
Usage: python scripts/progress.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import RUNS, load_yaml, run_id  # noqa: E402

grid, cfg = load_yaml("configs/grid.yaml"), load_yaml("configs/encoders.yaml")
secs = {}
remaining = {}
for m in cfg["order"]:
    done = 0
    for ts in grid["train_sets"]:
        for tc in grid["train_conditions"]:
            for s in grid["seeds"]:
                f = RUNS / run_id(m, ts, tc, s) / "run.json"
                if f.exists():
                    done += 1
                    j = json.loads(f.read_text())
                    secs.setdefault(ts, []).append(j["train_seconds"] + j["eval_seconds"])
                else:
                    remaining[ts] = remaining.get(ts, 0) + 1
    total = len(grid["train_sets"]) * len(grid["train_conditions"]) * len(grid["seeds"])
    print(f"{m:14s} {done:3d}/{total} done")
eta = sum(n * (sum(secs[ts]) / len(secs[ts])) for ts, n in remaining.items() if ts in secs)
unknown = [ts for ts in remaining if ts not in secs]
print(f"remaining runs: {sum(remaining.values())}; ETA ~ {eta / 3600:.1f} h"
      + (f" (+ runs on {unknown} not yet timed)" if unknown else ""))
