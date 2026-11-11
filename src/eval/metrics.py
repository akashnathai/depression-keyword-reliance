"""Metrics, bootstrap CIs, ECE and the rdc binary collapse."""
import numpy as np


def collapse_binary(probs: np.ndarray) -> np.ndarray:
    """3-class probabilities -> binary [P(none), P(moderate)+P(severe)] for rdc."""
    if probs.shape[1] == 2:
        return probs
    return np.stack([probs[:, 0], probs[:, 1:].sum(axis=1)], axis=1)


def confusion(y, p, k):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def macro_f1_batch(y, p, k, idx):
    """Macro-F1 for many bootstrap index sets at once. idx: (B, n)."""
    yb, pb = y[idx], p[idx]
    codes = yb * k + pb
    B = idx.shape[0]
    cms = np.zeros((B, k * k), dtype=np.int64)
    for c in range(k * k):
        cms[:, c] = (codes == c).sum(axis=1)
    cms = cms.reshape(B, k, k)
    tp = np.einsum("bii->bi", cms).astype(float)
    fp = cms.sum(axis=1) - tp
    fn = cms.sum(axis=2) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=1)


def ece(probs, y, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def compute_metrics(y, probs, k, n_boot=1000, seed=0, n_bins=15, chunk=200):
    """y: int labels in [0, k); probs: (n, k) probabilities (already collapsed if needed)."""
    y = np.asarray(y, dtype=np.int64)
    p = probs.argmax(axis=1)
    cm = confusion(y, p, k)
    f1 = per_class_f1(cm)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(k), where=support > 0)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([
        macro_f1_batch(y, p, k, rng.integers(0, len(y), size=(min(chunk, n_boot - i), len(y))))
        for i in range(0, n_boot, chunk)])
    out = {
        "n": len(y),
        "macro_f1": f1.mean(),
        "macro_f1_lo": np.percentile(boots, 2.5),
        "macro_f1_hi": np.percentile(boots, 97.5),
        "accuracy": (p == y).mean(),
        "weighted_f1": (f1 * support).sum() / support.sum(),
        "macro_recall": recall.mean(),
        "ece": ece(probs, y, n_bins),
    }
    for c in range(3):
        out[f"f1_c{c}"] = f1[c] if c < k else np.nan
    return out
