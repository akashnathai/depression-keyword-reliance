"""McNemar test and Holm correction."""
import numpy as np
from scipy.stats import binomtest, chi2


def mcnemar(y, pred_a, pred_b):
    """Paired test on correctness. Exact binomial if discordant < 25, else chi-squared with continuity correction."""
    a = np.asarray(pred_a) == np.asarray(y)
    b = np.asarray(pred_b) == np.asarray(y)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p": 1.0, "test": "none"}
    if n < 25:
        return {"n01": n01, "n10": n10, "stat": float(min(n01, n10)),
                "p": binomtest(min(n01, n10), n, 0.5).pvalue, "test": "exact"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"n01": n01, "n10": n10, "stat": stat, "p": float(chi2.sf(stat, 1)), "test": "chi2-cc"}


def holm(pvals):
    """Holm–Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def majority_vote(pred_matrix):
    """(n_seeds, n) int predictions -> (n,) majority; ties go to the lowest seed index's vote."""
    pm = np.asarray(pred_matrix)
    out = pm[0].copy()
    k = pm.max() + 1
    counts = np.stack([(pm == c).sum(axis=0) for c in range(k)])
    best = counts.max(axis=0)
    winner = counts.argmax(axis=0)
    unique_best = (counts == best).sum(axis=0) == 1
    out[unique_best] = winner[unique_best]
    return out
