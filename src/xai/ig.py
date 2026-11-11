"""Integrated Gradients on the embedding layer (captum LayerIntegratedGradients).

Baseline: pad tokens everywhere except the special start/end tokens. Target: predicted class
probability. Token attributions (summed over the embedding dimension) are summed into whitespace
words via the tokenizer's character offsets.
"""
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients

from src.xai.common import word_spans


def explain_ig(clf, text, target, n_steps=50, internal_batch_size=10):
    tok, model = clf.tok, clf.model
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=clf.max_len, return_tensors="pt")
    ids = enc["input_ids"].to(clf.device)
    mask = enc["attention_mask"].to(clf.device)
    offsets = enc["offset_mapping"][0].tolist()
    special = torch.tensor([i in tok.all_special_ids for i in ids[0].tolist()], device=clf.device)
    base = torch.where(special, ids[0], torch.full_like(ids[0], tok.pad_token_id))[None]

    def fwd(input_ids, attention_mask):
        return torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits, -1)

    emb = model.base_model.embeddings
    lig = LayerIntegratedGradients(fwd, emb)
    attr, delta = lig.attribute(ids, baselines=base, additional_forward_args=(mask,), target=int(target),
                                n_steps=n_steps, internal_batch_size=internal_batch_size,
                                return_convergence_delta=True)
    tok_attr = attr.sum(-1)[0].detach().float().cpu().numpy()

    spans = word_spans(text)
    starts = np.array([s for s, _ in spans])
    out = np.zeros(len(spans))
    for a, (s, e), sp in zip(tok_attr, offsets, special.tolist()):
        if sp or e <= s:
            continue
        j = np.searchsorted(starts, s, side="right") - 1
        if j >= 0 and s < spans[j][1]:
            out[j] += a
    return out, float(delta.abs().item())
