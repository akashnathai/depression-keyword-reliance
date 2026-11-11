"""Phase 2: run the tfidf-lr grid (train sets x train conditions x seeds), evaluate every run on
all test sets x test-time conditions, then aggregate into results/master_results.csv.
Resumable: a run whose 16 prediction files exist is skipped.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.aggregate import aggregate  # noqa: E402
from src.eval.evaluate import evaluate_run  # noqa: E402
from src.models.tfidf_lr import TfidfLR  # noqa: E402
from src.utils.io import RUNS, load_split, load_yaml, pred_path, run_id  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def main():
    grid, cfg = load_yaml("configs/grid.yaml"), load_yaml("configs/tfidf_lr.yaml")
    for ts in grid["train_sets"]:
        for tc in grid["train_conditions"]:
            for seed in grid["seeds"]:
                rid = run_id("tfidf-lr", ts, tc, seed)
                if all(pred_path(rid, s, c).exists() for s in grid["test_sets"]
                       for c in grid["test_conditions"]):
                    print("skip", rid)
                    continue
                t0 = time.time()
                tr, dv = load_split(ts, "train", tc), load_split(ts, "dev", tc)
                model = TfidfLR(cfg, seed).fit(tr["text"].tolist(), tr["label"], dv["text"].tolist(), dv["label"])
                evaluate_run(rid, model.predict_proba, grid["test_sets"], grid["test_conditions"])
                log = {"run_id": rid, "C": model.C, "val_macro_f1": model.val_f1,
                       "val_scores": model.val_scores, "fit_seconds": round(model.fit_seconds, 1),
                       "total_seconds": round(time.time() - t0, 1), "top_features": model.top_features()}
                (RUNS / rid).mkdir(parents=True, exist_ok=True)
                (RUNS / rid / "run.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
                print(f"{rid}: C={model.C} val F1={model.val_f1:.3f} ({log['total_seconds']}s)", flush=True)
    df, n = aggregate()
    print(f"aggregated {n} new files; master rows: {len(df)}")


if __name__ == "__main__":
    main()
