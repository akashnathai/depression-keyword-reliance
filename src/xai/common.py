"""Shared pieces for Phase 6: model wrapper, word segmentation, visible-text truncation.

All explainers and the faithfulness test use one word segmentation: whitespace-separated chunks
(`\\S+`). Long posts are first cut to the part the classifier actually sees (head 128 + tail 382
tokens, snapped to word boundaries), so every word that is explained is also visible to the model.
"""
import re

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.train_encoder import HeadTail, predict_proba

WORD_SPLIT = re.compile(r"\S+")


def word_spans(text):
    return [(m.start(), m.end()) for m in WORD_SPLIT.finditer(text)]


def words(text):
    return WORD_SPLIT.findall(text)


class Classifier:
    def __init__(self, path, max_len=512, head=128, batch_size=32, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.enc = HeadTail(self.tok, max_len, head)
        self.bs, self.device, self.max_len, self.head = batch_size, device, max_len, head

    def proba(self, texts):
        return predict_proba(self.model, self.enc, list(texts), self.bs, self.device)

    def visible_text(self, text):
        """Head+tail-truncated text, cut at word boundaries (unchanged if it already fits)."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        budget = self.enc.budget
        if len(off) <= budget:
            return text
        head_end = off[self.head - 1][1]
        tail_start = off[len(off) - (budget - self.head)][0]
        head_end = text.rfind(" ", 0, head_end + 1)  # drop a partly kept word
        nxt = text.find(" ", tail_start)
        tail_start = nxt + 1 if nxt != -1 and not text[tail_start - 1].isspace() else tail_start
        return (text[:max(head_end, 0)].rstrip() + " " + text[tail_start:].lstrip()).strip()


def remove_words(text, drop):
    """Rebuild text from whitespace words, leaving out the word indices in `drop` (a set)."""
    return " ".join(w for i, w in enumerate(words(text)) if i not in drop)


def keep_words(text, keep):
    return " ".join(w for i, w in enumerate(words(text)) if i in keep)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
