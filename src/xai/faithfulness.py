"""Comprehensiveness / sufficiency (DeYoung et al., 2020) and AOPC.

Words are ranked by attribution toward the predicted class (highest first). For k% of words:
  comprehensiveness = p(y|x) - p(y|x without top-k words)   (higher = more faithful)
  sufficiency       = p(y|x) - p(y|only top-k words)        (lower  = more faithful)
AOPC = mean over k. Perturbation = word deletion (independent of how the explainers mask).
"""
import numpy as np

from src.xai.common import keep_words, remove_words

KS = (1, 5, 10, 20, 50)


def top_sets(order, n):
    return [set(order[:max(1, int(round(k * n / 100)))].tolist()) for k in KS]


def faithfulness(clf, text, target, p_full, rankings):
    """rankings: {name: word index order (most important first)} -> {name: {comp_k, suff_k}}."""
    n = len(rankings[next(iter(rankings))])
    texts, keys = [], []
    for name, order in rankings.items():
        for k, s in zip(KS, top_sets(np.asarray(order), n)):
            texts += [remove_words(text, s), keep_words(text, s)]
            keys += [(name, "comp", k), (name, "suff", k)]
    probs = clf.proba(texts)[:, int(target)]
    out = {}
    for (name, kind, k), p in zip(keys, probs):
        out.setdefault(name, {})[f"{kind}_{k}"] = float(p_full - p)
    for name in out:
        out[name]["aopc_comp"] = float(np.mean([out[name][f"comp_{k}"] for k in KS]))
        out[name]["aopc_suff"] = float(np.mean([out[name][f"suff_{k}"] for k in KS]))
    return out


def rank(scores):
    return np.argsort(-np.asarray(scores), kind="stable")
