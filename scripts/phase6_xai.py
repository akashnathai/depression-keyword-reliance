"""Phase 6: explanations for the kept MentalRoBERTa models (orig vs mask-topic trained).

  python scripts/phase6_xai.py --explain     # IG + Partition SHAP + LIME + faithfulness, resumable
  python scripts/phase6_xai.py --summarise   # tables 5/6, figures, PRIMATE if available
  options: --n-per-class 100 (default), --limit N (quick test on N posts per model)

Per-post records go to results/xai/<tag>/records.jsonl (ids and scores only, no text; results/xai is
git-ignored anyway). Texts are never written to tables or figures; the example figure uses a
constructed sentence, not a real post.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.io import ROOT, RUNS, load_split  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
XAI, TAB, FIG = ROOT / "results/xai", ROOT / "results/tables", ROOT / "results/figures"
METHODS = ("ig", "shap", "lime")
MNAME = {"ig": "Integrated Gradients", "shap": "Partition SHAP", "lime": "LIME", "random": "Random"}
TAGS = ("orig", "mask-topic")
N_RANDOM = 5
SEED = 42

# Constructed (not taken from any dataset) sentence for the example figure; mixes topic and symptom words.
EXAMPLE = ("Started a new antidepressant last month and my therapist thinks it is helping, but I still "
           "cannot sleep, I feel exhausted all day and nothing seems worth the effort anymore.")


def kept_path(tag):
    info = json.loads((RUNS / f"kept__mentalroberta__depsign__{tag}.json").read_text())
    return RUNS / f"mentalroberta__depsign__{tag}__s{info['best_seed']}" / "best_model", info["best_seed"]


def sample(n_per_class):
    df = load_split("depsign", "test", "orig")
    parts = [g.sample(n=n_per_class, random_state=SEED) for _, g in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def post_rng(pid):
    return np.random.default_rng(int(hashlib.md5(f"{SEED}:{pid}".encode()).hexdigest()[:8], 16))


# ---------------------------------------------------------------- explain
def explain(n_per_class, limit):
    from src.xai.attribution_mass import mass, word_groups
    from src.xai.common import Classifier, set_seed, words
    from src.xai.faithfulness import faithfulness, rank
    from src.xai.ig import explain_ig
    from src.xai.lime_text import explain_lime, make_lime
    from src.xai.shap_partition import explain_shap, make_shap

    posts = sample(n_per_class)
    if limit:
        posts = posts.head(limit)
    for tag in TAGS:
        path, seed = kept_path(tag)
        out_dir = XAI / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_file = out_dir / "records.jsonl"
        done = {json.loads(line)["id"] for line in rec_file.open(encoding="utf-8")} if rec_file.exists() else set()
        todo = posts[~posts["id"].isin(done)]
        print(f"[{tag}] model seed {seed}: {len(done)} done, {len(todo)} to go", flush=True)
        if not len(todo):
            continue
        set_seed(SEED)
        clf = Classifier(path)
        shap_ex, lime_ex = make_shap(clf), make_lime(clf, SEED)
        t_all = time.time()
        with rec_file.open("a", encoding="utf-8") as fh:
            for n, row in enumerate(todo.itertuples(), 1):
                t0 = time.time()
                text = clf.visible_text(row.text)
                nw = len(words(text))
                probs = clf.proba([text])[0]
                pred = int(probs.argmax())
                scores = {}
                scores["ig"], delta = explain_ig(clf, text, pred)
                scores["shap"] = explain_shap(shap_ex, text, pred, nw)
                scores["lime"] = explain_lime(lime_ex, clf, text, pred, nw)
                rng = post_rng(row.id)
                rankings = {m: rank(scores[m]) for m in METHODS}
                rankings.update({f"random{i}": rng.permutation(nw) for i in range(N_RANDOM)})
                faith = faithfulness(clf, text, pred, float(probs[pred]), rankings)
                groups = word_groups(text)
                rec = {"id": row.id, "label": int(row.label), "pred": pred, "probs": probs.tolist(),
                       "n_words": nw, "truncated": text != row.text, "ig_delta": delta,
                       "scores": {m: np.round(scores[m], 6).tolist() for m in METHODS},
                       "groups": groups.tolist(), "faith": faith,
                       "mass": {m: mass(scores[m], groups) for m in METHODS}}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t_all
                    print(f"  [{tag}] {n}/{len(todo)}  last {time.time() - t0:.1f}s  "
                          f"eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
        del clf
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- summarise
def load_records(tag):
    f = XAI / tag / "records.jsonl"
    return [json.loads(line) for line in f.open(encoding="utf-8")] if f.exists() else []


def boot_ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def faith_table(recs):
    from src.xai.faithfulness import KS
    rows = []
    for tag, rs in recs.items():
        for m in list(METHODS) + ["random"]:
            row = {"model": tag, "method": m}
            for key in ["aopc_comp", "aopc_suff"] + [f"comp_{k}" for k in KS] + [f"suff_{k}" for k in KS]:
                if m == "random":
                    vals = [np.mean([r["faith"][f"random{i}"][key] for i in range(N_RANDOM)]) for r in rs]
                else:
                    vals = [r["faith"][m][key] for r in rs]
                mu, lo, hi = boot_ci(vals)
                row.update({key: mu, f"{key}_lo": lo, f"{key}_hi": hi})
            rows.append(row)
    return pd.DataFrame(rows)


def mass_table(recs):
    from src.xai.attribution_mass import GROUPS
    rows = []
    for tag, rs in recs.items():
        for m in METHODS:
            row = {"model": tag, "method": m, "n_posts": len(rs)}
            for g in GROUPS:
                sh = np.array([r["mass"][m][f"share_{g}"] for r in rs], dtype=float)
                ts = np.array([r["mass"][m][f"tokshare_{g}"] for r in rs], dtype=float)
                mu, lo, hi = boot_ci(sh)
                row.update({f"share_{g}": mu, f"share_{g}_lo": lo, f"share_{g}_hi": hi,
                            f"tokshare_{g}": float(np.nanmean(ts)),
                            f"norm_{g}": mu / float(np.nanmean(ts)) if np.nanmean(ts) > 0 else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def mass_diff(recs, group="topic"):
    """Paired bootstrap of share(mask-topic model) - share(orig model) on the same posts."""
    out = []
    a = {r["id"]: r for r in recs["orig"]}
    b = {r["id"]: r for r in recs["mask-topic"]}
    ids = sorted(set(a) & set(b))
    for m in METHODS:
        d = np.array([b[i]["mass"][m][f"share_{group}"] - a[i]["mass"][m][f"share_{group}"] for i in ids], dtype=float)
        mu, lo, hi = boot_ci(d)
        out.append({"method": m, "group": group, "diff_masktopic_minus_orig": mu, "lo": lo, "hi": hi, "n": len(ids)})
    return pd.DataFrame(out)


def write_tables(ft, mt, md, primate):
    from src.xai.faithfulness import KS  # noqa: F401
    TAB.mkdir(parents=True, exist_ok=True)
    ft.to_csv(TAB / "table5_faithfulness.csv", index=False, float_format="%.4f")
    mt.to_csv(TAB / "table6_attribution_mass.csv", index=False, float_format="%.4f")
    md.to_csv(TAB / "table6b_topic_mass_diff.csv", index=False, float_format="%.4f")

    def c(r, k):
        return f"{r[k]:.3f} \\tiny[{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"

    body = ("Model & Explainer & Compr.\\ AOPC $\\uparrow$ & Suff.\\ AOPC $\\downarrow$ & Compr.@20\\% & Suff.@20\\% \\\\\n"
            "\\midrule\n")
    for tag in TAGS:
        sub = ft[ft.model == tag]
        for i, r in enumerate(sub.itertuples()):
            r = r._asdict()
            body += (("MentalRoBERTa (" + tag + ")") if i == 0 else "") + f" & {MNAME[r['method']]} & " + \
                " & ".join(c(r, k) for k in ["aopc_comp", "aopc_suff", "comp_20", "suff_20"]) + " \\\\\n"
        body += "\\addlinespace[2pt]\n"
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Faithfulness on 300 DepSign test posts (100 per class), "
           "predicted class. Comprehensiveness: probability drop when the top-$k$\\% words are deleted (higher = more faithful); "
           "sufficiency: drop when only they are kept (lower = more faithful); AOPC averages $k\\in\\{1,5,10,20,50\\}$. "
           "Random: mean of 5 random rankings. Bootstrap 95\\% CI over posts.}\n\\label{tab:faith}\n\\scriptsize\n"
           "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llcccc}\n\\toprule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (TAB / "table5_faithfulness.tex").write_text(tex, encoding="utf-8")

    body = ("Model & Explainer & Topic & Suicide & Symptom & Other & Topic (norm.) & Symptom (norm.) \\\\\n\\midrule\n")
    for tag in TAGS:
        sub = mt[mt.model == tag]
        for i, r in enumerate(sub.itertuples()):
            r = r._asdict()
            body += (("MentalRoBERTa (" + tag + ")") if i == 0 else "") + f" & {MNAME[r['method']]} & " + \
                " & ".join(f"{r[f'share_{g}']:.3f}" for g in ["topic", "suicide", "symptom", "other"]) + \
                f" & {r['norm_topic']:.2f} & {r['norm_symptom']:.2f} \\\\\n"
        body += "\\addlinespace[2pt]\n"
    tok = mt[mt.model == "orig"].iloc[0]
    note = (f"Share of words in each group: topic {100 * tok.tokshare_topic:.1f}\\%, suicide {100 * tok.tokshare_suicide:.1f}\\%, "
            f"symptom {100 * tok.tokshare_symptom:.1f}\\%. ")
    if primate is None:
        note += "PRIMATE construct validity not computed (dataset access pending)."
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{Share of positive attribution per word group (mean over 300 posts). "
           "Norm.: share divided by the group's share of words (1 = proportional). " + note + "}\n\\label{tab:mass}\n"
           "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{llcccccc}\n\\toprule\n" + body +
           "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (TAB / "table6_attribution_mass.tex").write_text(tex, encoding="utf-8")


def figures(ft, mt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.xai.faithfulness import KS
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase5_report import AXIS, GRIDC, INK, INK2, SLOTS, save, style

    # faithfulness curves
    col = {"ig": SLOTS[0], "shap": SLOTS[1], "lime": SLOTS[2], "random": "#898781"}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharex=True)
    for j, (tag, kind) in enumerate([(t, k) for k in ("comp", "suff") for t in TAGS]):
        ax = axes[j]
        sub = ft[ft.model == tag].set_index("method")
        for m in list(METHODS) + ["random"]:
            ys = [sub.loc[m, f"{kind}_{k}"] for k in KS]
            ax.plot(KS, ys, color=col[m], linewidth=2, marker="o", markersize=4,
                    linestyle="--" if m == "random" else "-", label=MNAME[m])
        ax.set_xscale("log")
        ax.set_xticks(KS, [str(k) for k in KS])
        ax.set_title(f"{'Compr.' if kind == 'comp' else 'Suff.'} · {tag}", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("% words", fontsize=7, color=INK2)
        style(ax)
    axes[0].set_ylabel("Δ p(predicted class)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.35))
    save(fig, "faithfulness_curves")

    # attribution-mass stacked bars
    gcol = {"topic": SLOTS[1], "suicide": SLOTS[7], "symptom": SLOTS[2], "other": "#c3c2b7"}
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    labels, x = [], 0
    for m in METHODS:
        for tag in TAGS:
            r = mt[(mt.model == tag) & (mt.method == m)].iloc[0]
            bottom = 0
            for g in ("topic", "suicide", "symptom", "other"):
                ax.barh(x, r[f"share_{g}"], left=bottom, color=gcol[g], height=0.7,
                        edgecolor="#fcfcfb", linewidth=1.5, label=g if x == 0 else None, zorder=3)
                bottom += r[f"share_{g}"]
            labels.append(f"{MNAME[m]} · {tag}")
            x += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of positive attribution", fontsize=8, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=7, colors=INK2)
    ax.grid(axis="x", color=GRIDC, linewidth=0.6, zorder=0)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.2))
    save(fig, "attribution_mass")


def example_figure():
    """IG on a constructed sentence for both models (no real post is shown)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.colors import LinearSegmentedColormap
    from src.xai.common import Classifier, words
    from src.xai.ig import explain_ig
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase5_report import INK, save

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])  # blue - gray - red
    ws = words(EXAMPLE)
    rows = []
    for tag in TAGS:
        clf = Classifier(kept_path(tag)[0], device="cpu")  # GPU may be busy with training
        p = clf.proba([EXAMPLE])[0]
        a, _ = explain_ig(clf, EXAMPLE, int(p.argmax()))
        rows.append((tag, int(p.argmax()), a / (np.abs(a).max() + 1e-12)))
        del clf
        torch.cuda.empty_cache()
    fig = plt.figure(figsize=(7.2, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    cls = ["not depressed", "moderate", "severe"]
    y = 0.92
    for tag, pred, a in rows:
        ax.text(0.01, y, f"MentalRoBERTa ({tag}) → predicted: {cls[pred]}", fontsize=8, color=INK, va="top",
                transform=ax.transAxes, weight="bold")
        y -= 0.13
        x = 0.01
        for w, v in zip(ws, a):
            t = ax.text(x, y, w, fontsize=7.5, va="top", transform=ax.transAxes, color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc=cmap((v + 1) / 2), ec="none"))
            fig.canvas.draw()
            bb = t.get_bbox_patch().get_window_extent().transformed(ax.transAxes.inverted())
            x = bb.x1 + 0.008
            if x > 0.93:
                x, y = 0.01, y - 0.12
        y -= 0.17
    ax.text(0.01, 0.02, "Constructed example sentence (not from any dataset). Red: supports the predicted class; "
            "blue: against. Integrated Gradients, scaled per model.", fontsize=6.5, color="#52514e",
            transform=ax.transAxes)
    save(fig, "attribution_example")


def primate_validity():
    """Placeholder until PRIMATE is obtained through the official agreement."""
    raw = ROOT / "data/raw/primate"
    if not raw.exists() or not any(raw.iterdir()):
        print("PRIMATE not available: construct-validity analysis skipped.")
        return None
    raise NotImplementedError("PRIMATE files found: write the loader for the delivered format, then rerun.")


def summarise():
    recs = {tag: load_records(tag) for tag in TAGS}
    print({t: len(r) for t, r in recs.items()})
    ft, mt = faith_table(recs), mass_table(recs)
    md = pd.concat([mass_diff(recs, "topic"), mass_diff(recs, "symptom")])
    primate = primate_validity()
    write_tables(ft, mt, md, primate)
    figures(ft, mt)
    example_figure()
    stats = {t: {"n": len(r), "acc": float(np.mean([x["pred"] == x["label"] for x in r])),
                 "truncated_share": float(np.mean([x["truncated"] for x in r])),
                 "ig_delta_median": float(np.median([x["ig_delta"] for x in r]))} for t, r in recs.items()}
    (TAB / "xai_sample_stats.json").write_text(json.dumps(stats, indent=1))
    pd.set_option("display.width", 200)
    print(ft[["model", "method", "aopc_comp", "aopc_suff"]].round(3).to_string())
    print(mt[["model", "method", "share_topic", "share_symptom", "norm_topic", "norm_symptom"]].round(3).to_string())
    print(md.round(3).to_string())
    print(stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.explain:
        import torch
        torch.cuda.set_per_process_memory_fraction(0.85)
        explain(a.n_per_class, a.limit)
    if a.summarise:
        summarise()


if __name__ == "__main__":
    main()
