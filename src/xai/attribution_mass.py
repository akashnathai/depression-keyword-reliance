"""Share of positive attribution on topic / suicide / symptom / other words.

A word belongs to a lexicon group if its character span overlaps a lexicon match
(priority topic > suicide > symptom). Normalised share = share / group's share of words.
"""
import numpy as np

from src.data.lexicon import compile_lexicon, find_spans
from src.xai.common import word_spans

GROUPS = ("topic", "suicide", "symptom", "other")
_PAT = {}


def word_groups(text):
    if not _PAT:
        for g in GROUPS[:3]:
            _PAT[g] = compile_lexicon([g])
    spans = word_spans(text)
    lab = np.array(["other"] * len(spans), dtype=object)
    for g in reversed(GROUPS[:3]):  # lowest priority first, so topic overwrites
        for s, e in find_spans(text, _PAT[g]):
            for i, (ws, we) in enumerate(spans):
                if ws < e and s < we:
                    lab[i] = g
    return lab


def mass(scores, groups):
    pos = np.clip(np.asarray(scores, dtype=float), 0, None)
    tot = pos.sum()
    out = {}
    for g in GROUPS:
        m = groups == g
        out[f"share_{g}"] = float(pos[m].sum() / tot) if tot > 0 else np.nan
        out[f"tokshare_{g}"] = float(m.mean())
    return out
