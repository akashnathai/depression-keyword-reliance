"""Lexicon prevalence check, run before creating the masked data.

Writes data/processed/lexicon_preview.md: % posts with at least one match per lexicon, per mapping and
class (all splits pooled), and the most frequent matched surface forms. No post text is printed.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labels import CLASS_NAMES  # noqa: E402
from src.data.lexicon import NAMES, compile_lexicon  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
P = Path("data/processed")
pats = {n: compile_lexicon([n]) for n in NAMES}
lines = ["# Lexicon preview", "", "% of posts with at least one match (train+dev+test pooled).", "",
         "| mapping | class | n | topic % | suicide % | symptom % |", "|---|---|---|---|---|---|"]
forms = {n: Counter() for n in NAMES}
for mapping in ("depsign", "dsl3", "rdc"):
    df = pd.concat([pd.read_parquet(f) for f in P.glob(f"{mapping}_*.parquet")])
    hits = {n: df["text"].map(lambda t, p=pats[n]: p.findall(t)) for n in NAMES}
    for n in NAMES:
        for h in hits[n]:
            forms[n].update(x.lower() for x in h)
    for c, cname in enumerate(CLASS_NAMES[mapping]):
        m = df["label"] == c
        row = [f"{100 * hits[n][m].map(bool).mean():.1f}" for n in NAMES]
        lines.append(f"| {mapping} | {c} {cname} | {m.sum()} | " + " | ".join(row) + " |")
lines.append("")
for n in NAMES:
    lines.append(f"**Top matched forms — {n}:** " +
                 ", ".join(f"{w} ({k})" for w, k in forms[n].most_common(25)))
    lines.append("")
(P / "lexicon_preview.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
