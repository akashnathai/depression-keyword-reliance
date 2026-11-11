"""Partition SHAP over whitespace words (shap.Explainer + Text masker).

Masked words are replaced by the model's mask token (<mask> for RoBERTa). max_evals 500.
"""
import re

import numpy as np
import shap
from shap.maskers._text import SimpleTokenizer



def make_shap(clf):
    mask = clf.tok.mask_token
    fix = re.compile(rf"(?<={re.escape(mask)})(?=\S)")  # shap drops the space after a mask: "<mask>word"

    def f(xs):
        return clf.proba([fix.sub(" ", str(x)) for x in xs])

    masker = shap.maskers.Text(SimpleTokenizer(r"\s+"), mask_token=mask, collapse_mask_token=False)
    return shap.Explainer(f, masker, algorithm="partition")


def explain_shap(explainer, text, target, n_words, max_evals=500, batch_size=64):
    sv = explainer([text], max_evals=max_evals, batch_size=batch_size, silent=True)
    vals = np.asarray(sv.values[0])[:, int(target)]
    assert len(vals) == n_words, f"shap words {len(vals)} != {n_words}"
    return vals
