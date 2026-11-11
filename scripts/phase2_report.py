"""Print in-domain, cross-dataset and keyword-reliance-gap numbers for a model from master_results.csv.
Usage: python scripts/phase2_report.py [model]   (default tfidf-lr)
Also writes results/tables/summary_<model>.csv and krg_<model>.csv.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.aggregate import MASTER, krg_table, seed_summary  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)
model = sys.argv[1] if len(sys.argv) > 1 else "tfidf-lr"
m = pd.read_csv(MASTER)
m = m[m["model"] == model]
s = seed_summary(m)
k = krg_table(s)
Path("results/tables").mkdir(parents=True, exist_ok=True)
s.to_csv(f"results/tables/summary_{model}.csv", index=False, float_format="%.4f")
k.to_csv(f"results/tables/krg_{model}.csv", index=False, float_format="%.4f")

fmt = lambda r: f"{r.macro_f1:.3f} [{r.lo:.3f}, {r.hi:.3f}]"  # noqa: E731
print(f"=== {model}: IN-DOMAIN macro-F1 (orig test), mean over seeds [mean bootstrap 95% CI]")
ind = s[(s.train_set == s.test_set) & (s.test_cond == "orig")]
for _, r in ind.iterrows():
    print(f"  train {r.train_set:8s} {r.train_cond:15s} {fmt(r)}  sd={r.macro_f1_sd:.4f}  ECE={r.ece:.3f}")

print(f"\n=== {model}: CROSS-DATASET macro-F1 (orig test; rdc binary-collapsed)")
cross = s[(s.test_cond == "orig")].pivot_table(index=["train_set", "train_cond"], columns="test_set",
                                               values="macro_f1").round(3)
print(cross.to_string())

print(f"\n=== {model}: TEST-TIME MASKING (orig-trained). drop = F1(orig) - F1(masked test); KRG = F1(mask-random) - F1(mask-topic)")
print(k[["train_set", "test_set", "orig", "mask-topic", "mask-topic+sui", "mask-random",
         "drop_topic", "drop_topic_sui", "drop_random", "KRG"]].round(3).to_string(index=False))
