"""LIME over whitespace words (position-aware, bow=False). 1,000 samples.

Removed words are replaced by the model's mask token, as for SHAP.
"""
import numpy as np
from lime.lime_text import LimeTextExplainer


def make_lime(clf, seed):
    return LimeTextExplainer(class_names=["0", "1", "2"], split_expression=r"\s+", bow=False,
                             mask_string=clf.tok.mask_token, random_state=seed)


def explain_lime(explainer, clf, text, target, n_words, num_samples=1000):
    exp = explainer.explain_instance(text, clf.proba, labels=[int(target)], num_features=n_words,
                                     num_samples=num_samples)
    assert exp.domain_mapper.indexed_string.num_words() == n_words
    out = np.zeros(n_words)
    for idx, w in exp.as_map()[int(target)]:
        out[idx] = w
    return out
