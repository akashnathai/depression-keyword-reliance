"""Cross-dataset and test-time-masking evaluation of one trained model.

`predict_proba(texts) -> (n, 3) array` is supplied by the model. Writes one parquet per
(test set, test condition) with id, label, raw 3-class probabilities.
"""
import numpy as np
import pandas as pd

from src.utils.io import load_split, pred_path


def evaluate_run(rid, predict_proba, test_sets, test_conditions, overwrite=False):
    written = 0
    for ts in test_sets:
        for tc in test_conditions:
            out = pred_path(rid, ts, tc)
            if out.exists() and not overwrite:
                continue
            df = load_split(ts, "test", tc)
            probs = np.asarray(predict_proba(df["text"].tolist()), dtype=float)
            res = pd.DataFrame({"id": df["id"], "label": df["label"].astype(int)})
            for c in range(probs.shape[1]):
                res[f"p{c}"] = probs[:, c]
            out.parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(out, index=False)
            written += 1
    return written
