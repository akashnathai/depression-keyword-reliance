"""Phase 4: Qwen2.5-1.5B prompting and QLoRA.

  python scripts/phase4_llm.py --prompt          # zero-shot + 5-shot (examples from depsign / dsl3)
  python scripts/phase4_llm.py --qlora-timing    # one QLoRA run: depsign, orig, seed 13
  python scripts/phase4_llm.py --qlora           # QLoRA grid (resumable)
  options: --seeds 13 (restrict QLoRA seeds, e.g. if one run takes > 2 h)
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.aggregate import aggregate  # noqa: E402
from src.utils.io import N_CLASSES, RUNS, load_yaml, run_id  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def prompting(cfg, grid):
    from src.models.llm_prompt import Prompter, pick_shots, run_prompting
    pr = Prompter(cfg)
    seed = cfg["prompt_seed"]
    jobs = [("qwen-zs", "none", [], cfg["prompt_batch_size"]["zs"])]
    for ts in grid["train_sets"]:
        jobs.append(("qwen-fs", ts, pick_shots(ts, cfg, seed), cfg["prompt_batch_size"]["fs"]))
    for model, ts, shots, bs in jobs:
        rid = run_id(model, ts, "orig", seed)
        t0 = time.time()
        stats = run_prompting(pr, rid, shots, grid["test_sets"], grid["test_conditions"], bs, N_CLASSES)
        f = RUNS / rid / "run.json"
        old = json.loads(f.read_text())["stats"] if f.exists() else []
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"run_id": rid, "n_shots": len(shots), "shot_labels": [c for _, c in shots],
                                 "stats": old + stats, "seconds": round(time.time() - t0, 1)}, indent=1))
        aggregate()


def qlora(cfg, grid, timing, seeds):
    from src.models.train_encoder import run_done, train_run
    if timing:
        jobs = [("depsign", "orig", 13)]
    else:
        jobs = [(ts, tc, s) for ts in cfg["qlora_train_sets"] for tc in cfg["qlora_conditions"]
                for s in (seeds or grid["seeds"])]
    for n, (ts, tc, s) in enumerate(jobs, 1):
        rid = run_id("qwen-qlora", ts, tc, s)
        if run_done(rid, grid["test_sets"], grid["test_conditions"]):
            print(f"[{n}/{len(jobs)}] skip {rid}", flush=True)
            continue
        print(f"[{n}/{len(jobs)}] start {rid}  {time.strftime('%H:%M:%S')}", flush=True)
        try:
            log = train_run(rid, "qwen-qlora", cfg, ts, tc, s, grid["test_sets"], grid["test_conditions"])
        except Exception:
            traceback.print_exc()
            print(f"[{n}/{len(jobs)}] FAILED {rid}", flush=True)
            continue
        print(f"[{n}/{len(jobs)}] done {rid}: val F1={log['val_macro_f1']:.4f} best epoch={log['best_epoch']} "
              f"train={log['train_seconds'] / 60:.1f}min eval={log['eval_seconds'] / 60:.1f}min "
              f"peak VRAM={log['peak_vram_gb']}GB", flush=True)
        aggregate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="store_true")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--qlora-timing", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    cfg, grid = load_yaml("configs/qwen.yaml"), load_yaml("configs/grid.yaml")
    import torch
    # Windows (WDDM) silently spills to shared RAM instead of raising OOM, which makes runs crawl.
    # Capping the allocator makes it free its cache and retry before that happens.
    torch.cuda.set_per_process_memory_fraction(cfg.get("vram_fraction", 0.85))
    if a.prompt:
        prompting(cfg, grid)
    if a.qlora or a.qlora_timing:
        qlora(cfg, grid, a.qlora_timing, a.seeds)


if __name__ == "__main__":
    main()
