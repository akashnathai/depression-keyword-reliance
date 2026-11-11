"""Phase 5, additional analyses reported in the paper (computed from saved data and predictions).

  python scripts/phase5_extra.py

1. placeholder_count_baseline.csv: how much the number of [KW] placeholders alone predicts the label
   (logistic regression on the count; trained on train, scored on test), next to dummy baselines.
2. krg_pretraining_diff.csv: paired bootstrap of KRG(domain-pretrained) - KRG(general) on the same
   test posts and resamples (MentalRoBERTa vs RoBERTa, MentalBERT vs BERT).
3. attribution_ratio.csv: share of positive attribution / share of words per word group, with a
   bootstrap CI over posts (ratio of means), for the attribution-ratio figure.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_report import B, GRID, TAB, boot_idx, f1_point, load_pred, seeds_of  # noqa: E402
from src.eval.aggregate import MASTER  # noqa: E402
from src.eval.metrics import macro_f1_batch  # noqa: E402
from src.utils.io import N_CLASSES, ROOT, load_split  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PH = "[KW]"


def boot_f1(y, p, k):
    idx = boot_idx(len(y))
    b = np.concatenate([macro_f1_batch(y, p, k, idx[i:i + 200]) for i in range(0, B, 200)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def placeholder_baseline():
    rows = []
    for ts in GRID["train_sets"]:
        for cond in ["mask-topic", "mask-topic+sui"]:
            tr, te = load_split(ts, "train", cond), load_split(ts, "test", cond)
            xtr = tr["text"].str.count(re.escape(PH)).to_numpy()[:, None]
            xte = te["text"].str.count(re.escape(PH)).to_numpy()[:, None]
            ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy().astype(np.int64)
            k = N_CLASSES[ts]
            clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(xtr, ytr)
            p = clf.predict(xte).astype(np.int64)
            lo, hi = boot_f1(yte, p, k)
            strat = np.mean([f1_point(yte, DummyClassifier(strategy="stratified", random_state=s)
                                      .fit(xtr, ytr).predict(xte).astype(np.int64), k) for s in range(200)])
            unif = np.mean([f1_point(yte, DummyClassifier(strategy="uniform", random_state=s)
                                     .fit(xtr, ytr).predict(xte).astype(np.int64), k) for s in range(200)])
            rows.append({"train_test_set": ts, "condition": cond, "macro_f1_count_only": f1_point(yte, p, k),
                         "ci_lo": lo, "ci_hi": hi, "dummy_stratified_mean": strat, "dummy_uniform_mean": unif,
                         "share_posts_with_kw_test": float((xte[:, 0] > 0).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "placeholder_count_baseline.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string())


def krg_seed_boot(rids, te, idx, ids, y, k):
    """(point, bootstrap vector) of seed-mean F1(mask-random) - F1(mask-topic)."""
    pts, boots = [], []
    for rid in rids:
        _, _, pr = load_pred(rid, te, "mask-random", ids)
        _, _, pt = load_pred(rid, te, "mask-topic", ids)
        pts.append(f1_point(y, pr, k) - f1_point(y, pt, k))
        boots.append(np.concatenate([macro_f1_batch(y, pr, k, idx[i:i + 200]) - macro_f1_batch(y, pt, k, idx[i:i + 200])
                                     for i in range(0, B, 200)]))
    return float(np.mean(pts)), np.mean(boots, axis=0)


def krg_pretraining_diff():
    master = pd.read_csv(MASTER)
    pairs = [("mentalroberta", "roberta"), ("mentalbert", "bert")]
    settings = [("depsign", "depsign"), ("depsign", "rdc"), ("dsl3", "dsl3"), ("dsl3", "rdc")]
    rows = []
    for a, b in pairs:
        for tr, te in settings:
            ra, rb = seeds_of(master, a, tr, "orig"), seeds_of(master, b, tr, "orig")
            ids, y, _ = load_pred(ra[0], te, "orig")
            k = N_CLASSES[te]
            idx = boot_idx(len(y))
            ka, ba = krg_seed_boot(ra, te, idx, ids, y, k)
            kb, bb = krg_seed_boot(rb, te, idx, ids, y, k)
            d = ba - bb
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            rows.append({"model_A": a, "model_B": b, "train_set": tr, "test_set": te, "KRG_A": ka, "KRG_B": kb,
                         "diff_A_minus_B": ka - kb, "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": lo > 0 or hi < 0})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "krg_pretraining_diff.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string())


def attribution_ratio():
    rows = []
    rng = np.random.default_rng(0)
    for tag in ("orig", "mask-topic"):
        recs = [json.loads(line) for line in (ROOT / "results/xai" / tag / "records.jsonl").open(encoding="utf-8")]
        idx = rng.integers(0, len(recs), size=(B, len(recs)))
        for m in ("ig", "shap", "lime"):
            for g in ("topic", "suicide", "symptom", "other"):
                sh = np.array([r["mass"][m][f"share_{g}"] for r in recs], dtype=float)
                ts = np.array([r["mass"][m][f"tokshare_{g}"] for r in recs], dtype=float)
                ok = ~np.isnan(sh)
                sh, ts = np.where(ok, sh, 0.0), np.where(ok, ts, 0.0)
                w = ok.astype(float)
                ratio = (sh.sum() / w.sum()) / (ts.sum() / w.sum())
                bs = (sh[idx].sum(1) / w[idx].sum(1)) / (ts[idx].sum(1) / w[idx].sum(1))
                rows.append({"model": tag, "method": m, "group": g, "ratio": ratio,
                             "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                             "attr_share": float(sh.sum() / w.sum()), "word_share": float(ts.sum() / w.sum())})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "attribution_ratio.csv", index=False, float_format="%.4f")
    print(t[t.group.isin(["topic", "symptom"])].round(3).to_string())


if __name__ == "__main__":
    print("== placeholder-count baseline"); placeholder_baseline()
    print("== KRG difference, domain-pretrained vs general"); krg_pretraining_diff()
    print("== attribution ratio"); attribution_ratio()
    print("== training-condition differences"); train_cond_diff()
    print("== prompted prediction shares"); prompt_shares()
    print("== per-seed KRG"); krg_per_seed()
    print("== DepSign raw counts"); depsign_counts()


def train_cond_diff():
    """Paired bootstrap of seed-mean macro-F1 differences between training conditions (orig test set).
    A minus B for (mask-topic vs orig) and (mask-random vs mask-topic); same posts and resamples."""
    master = pd.read_csv(MASTER)
    rows = []
    for model in sorted(master.model.unique()):
        if model in ("qwen-zs", "qwen-fs"):
            continue
        for tr in GRID["train_sets"]:
            for a, b in [("mask-topic", "orig"), ("mask-random", "mask-topic"), ("mask-random", "orig")]:
                ra, rb = seeds_of(master, model, tr, a), seeds_of(master, model, tr, b)
                if not ra or not rb:
                    continue
                for te in GRID["test_sets"]:
                    ids, y, _ = load_pred(rb[0], te, "orig")
                    k = N_CLASSES[te]
                    idx = boot_idx(len(y))

                    def seed_mean(rids):
                        pts, bs = [], []
                        for rid in rids:
                            _, _, p = load_pred(rid, te, "orig", ids)
                            pts.append(f1_point(y, p, k))
                            bs.append(np.concatenate([macro_f1_batch(y, p, k, idx[i:i + 200]) for i in range(0, B, 200)]))
                        return float(np.mean(pts)), np.mean(bs, axis=0)

                    fa, ba = seed_mean(ra)
                    fb, bb = seed_mean(rb)
                    d = ba - bb
                    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
                    rows.append({"model": model, "train_set": tr, "A": a, "B": b, "test_set": te,
                                 "f1_A": fa, "f1_B": fb, "diff_A_minus_B": fa - fb, "ci_lo": lo, "ci_hi": hi,
                                 "ci_excludes_0": lo > 0 or hi < 0, "n_seeds_A": len(ra), "n_seeds_B": len(rb)})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "train_cond_diff.csv", index=False, float_format="%.4f")
    print(t[t.A == "mask-topic"].round(3).to_string())


def prompt_shares():
    """Share of each predicted class for the prompted models on every orig test set."""
    rows = []
    for d in sorted((ROOT / "results/preds").glob("qwen-[zf]s__*")):
        for f in sorted(d.glob("*__orig.parquet")):
            p = pd.read_parquet(f)[["p0", "p1", "p2"]].to_numpy().argmax(1)
            rows.append({"run_id": d.name, "test_set": f.stem.split("__")[0], "n": len(p),
                         "share_none": float((p == 0).mean()), "share_moderate": float((p == 1).mean()),
                         "share_severe": float((p == 2).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "prompt_pred_shares.csv", index=False, float_format="%.4f")
    print(t.round(3).to_string())


def krg_per_seed():
    """KRG of every orig-trained seed (from master_results.csv), to show seed consistency."""
    m = pd.read_csv(MASTER)
    m = m[m.train_cond == "orig"]
    p = m.pivot_table(index=["model", "train_set", "seed", "test_set"], columns="test_cond", values="macro_f1").reset_index()
    p["KRG"] = p["mask-random"] - p["mask-topic"]
    p[["model", "train_set", "seed", "test_set", "KRG"]].to_csv(TAB / "krg_per_seed.csv", index=False, float_format="%.4f")
    q = p[(p.train_set == "depsign") & (p.test_set == "depsign") & p.model.isin(["roberta", "mentalroberta", "bert", "mentalbert"])]
    print(q.groupby("model").KRG.agg(["min", "max"]).round(3))


def depsign_counts():
    """Raw DepSign counts used in the Data section (duplicates in train; test posts removed by cleaning)."""
    from src.data.clean import clean_text, load_depsign, n_words
    raw = load_depsign()
    tr, te = raw[raw.split == "train"], raw[raw.split == "test"]
    te_clean = te.assign(text=te.text.map(clean_text))
    te_short = int((te_clean.text.map(n_words) < 5).sum())
    final_test = len(load_split("depsign", "test", "orig"))
    out = {"train_rows_raw": len(tr), "train_distinct_texts_raw": int(tr.text.nunique()),
           "test_rows_raw": len(te), "test_removed_short": te_short,
           "test_after_cleaning": len(te) - te_short, "test_final": final_test,
           "test_removed_total": len(te) - final_test,
           "test_removed_duplicates": len(te) - te_short - final_test}
    (TAB / "depsign_counts.json").write_text(json.dumps(out, indent=1))
    print(out)
