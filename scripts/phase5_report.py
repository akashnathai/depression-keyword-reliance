"""Phase 5: statistics, paper tables 2/3/4/7 and figures from results/.

  python scripts/phase5_report.py

Inputs : results/master_results.csv, results/preds/, results/runs/*/run.json
Outputs: results/tables/table{2,3,4,7}_*.{tex,csv}, mcnemar_holm.csv, key_numbers.json
         results/figures/{heatmap_<train>,krg_bars,transfer_drop,rdc_transfer}.{pdf,png}
All numbers are computed from the saved predictions.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.aggregate import MASTER, seed_summary  # noqa: E402
from src.eval.metrics import collapse_binary, macro_f1_batch  # noqa: E402
from src.eval.stats import holm, majority_vote, mcnemar  # noqa: E402
from src.utils.io import N_CLASSES, PREDS, ROOT, RUNS, load_yaml, pred_path  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
GRID = load_yaml("configs/grid.yaml")
B, BSEED = GRID["bootstrap"]["n"], GRID["bootstrap"]["seed"]

MODELS = ["tfidf-lr", "bert", "roberta", "mentalbert", "mentalroberta", "qwen-zs", "qwen-fs", "qwen-qlora"]
NAME = {"tfidf-lr": "TF-IDF+LR", "bert": "BERT", "roberta": "RoBERTa", "mentalbert": "MentalBERT",
        "mentalroberta": "MentalRoBERTa", "qwen-zs": "Qwen-ZS", "qwen-fs": "Qwen-FS", "qwen-qlora": "Qwen-QLoRA"}
TRAINED = ["tfidf-lr", "bert", "roberta", "mentalbert", "mentalroberta", "qwen-qlora"]
CONDS = ["orig", "mask-topic", "mask-topic+sui", "mask-random"]
TS_NAME = {"depsign": "DepSign", "dsl3": "DSD-3", "dsl3m": "DSD-3m", "rdc": "RDC"}
OTHER = {"depsign": "dsl3", "dsl3": "depsign"}  # 3-class cross-dataset target
SETTINGS = [("depsign", "depsign"), ("depsign", "rdc"), ("dsl3", "dsl3"), ("dsl3", "rdc")]

# fixed colours, so each model / condition keeps the same colour in every figure
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOR = {m: SLOTS[i] for i, m in enumerate(MODELS)}
COND_COLOR = {c: SLOTS[i] for i, c in enumerate(CONDS)}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, MUTED, GRIDC, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------- prediction helpers
def seeds_of(master, model, ts, tc):
    m = master[(master.model == model) & (master.train_set == ts) & (master.train_cond == tc)]
    return sorted(m["run_id"].unique())


def load_pred(rid, test_set, test_cond, order=None):
    """-> (ids, y, hard predictions); rdc: binary collapse P(0) vs P(1)+P(2), as in aggregate.py."""
    df = pd.read_parquet(pred_path(rid, test_set, test_cond))
    if order is not None:
        df = df.set_index("id").loc[order].reset_index()
    probs = df[sorted(c for c in df.columns if c.startswith("p") and c[1:].isdigit())].to_numpy()
    if N_CLASSES[test_set] == 2:
        probs = collapse_binary(probs)
    return df["id"].to_numpy(), df["label"].to_numpy().astype(np.int64), probs.argmax(1)


def boot_idx(n):
    return np.random.default_rng(BSEED).integers(0, n, size=(B, n))


def f1_point(y, p, k):
    return float(macro_f1_batch(y, p, k, np.arange(len(y))[None])[0])


def paired_diff(rids, test_set, cond_a, cond_b):
    """Seed-mean of F1(cond_a test) - F1(cond_b test), paired bootstrap CI (same resamples)."""
    k = N_CLASSES[test_set]
    ids, y, _ = load_pred(rids[0], test_set, "orig")
    idx = boot_idx(len(y))
    pts, boots = [], []
    for rid in rids:
        _, _, pa = load_pred(rid, test_set, cond_a, ids)
        _, _, pb = load_pred(rid, test_set, cond_b, ids)
        pts.append(f1_point(y, pa, k) - f1_point(y, pb, k))
        boots.append(np.concatenate([macro_f1_batch(y, pa, k, idx[i:i + 200]) - macro_f1_batch(y, pb, k, idx[i:i + 200])
                                     for i in range(0, B, 200)]))
    bm = np.mean(boots, axis=0)
    return float(np.mean(pts)), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def binary_in_domain(rids, train_set):
    """In-domain macro-F1 after the same binary collapse used for rdc (for a like-for-like rdc drop)."""
    vals = []
    for rid in rids:
        df = pd.read_parquet(pred_path(rid, train_set, "orig"))
        probs = collapse_binary(df[["p0", "p1", "p2"]].to_numpy())
        y = (df["label"].to_numpy() >= 1).astype(np.int64)
        vals.append(f1_point(y, probs.argmax(1), 2))
    return float(np.mean(vals))


# ---------------------------------------------------------------- LaTeX helpers
def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def write_tex(name, body, caption, label, colspec):
    tex = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
           "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{" + colspec + "}\n\\toprule\n"
           + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (TAB / f"{name}.tex").write_text(tex, encoding="utf-8")


def present(master, models):
    return [m for m in models if m in set(master.model)]


# ---------------------------------------------------------------- Table 2: in-domain
def table2(master, summ):
    rows = []
    for m in present(master, TRAINED):
        for ts in GRID["train_sets"]:
            for tc in CONDS:
                r = summ[(summ.model == m) & (summ.train_set == ts) & (summ.train_cond == tc)
                         & (summ.test_set == ts) & (summ.test_cond == "orig")]
                if len(r):
                    r = r.iloc[0]
                    rows.append({"model": m, "train_set": ts, "train_cond": tc, "n_seeds": r.n_seeds,
                                 "macro_f1": r.macro_f1, "sd": r.macro_f1_sd, "lo": r.lo, "hi": r.hi, "ece": r.ece})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table2_in_domain.csv", index=False, float_format="%.4f")
    body = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{train {TS_NAME[ts]}}}" for ts in GRID["train_sets"]) + " \\\\\n"
    body += "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nModel & " + " & ".join(
        ["orig", "m-topic", "m-t+sui", "m-random"] * 2) + " \\\\\n\\midrule\n"
    for m in t.model.unique():
        cells = []
        for ts in GRID["train_sets"]:
            sub = t[(t.model == m) & (t.train_set == ts)].set_index("train_cond")
            best = sub.macro_f1.max() if len(sub) else np.nan
            for tc in CONDS:
                if tc not in sub.index:
                    cells.append("--")
                    continue
                r = sub.loc[tc]
                v = f"{r.macro_f1:.3f}"
                v = f"\\textbf{{{v}}}" if np.isclose(r.macro_f1, best) else v
                sd = f"$\\pm${r.sd:.3f}" if r.n_seeds > 1 and not pd.isna(r.sd) else ""
                cells.append(f"\\shortstack{{{v}{sd}\\\\[-1pt]\\tiny[{r.lo:.3f}, {r.hi:.3f}]}}")
        body += NAME[m] + " & " + " & ".join(cells) + " \\\\\n"
    write_tex("table2_in_domain", body,
              "In-domain macro-F1 on the unmasked test set for every model and training condition. "
              "Mean $\\pm$ s.d.\\ over seeds; below: mean bootstrap 95\\% CI. Bold: best training condition per row and train set.",
              "tab:indomain", "l" + "c" * 8)
    return t


# ---------------------------------------------------------------- Table 3: cross-dataset + transfer drop
def table3(master, summ):
    rows = []
    for m in present(master, TRAINED):
        for ts in GRID["train_sets"]:
            for tc in CONDS:
                rids = seeds_of(master, m, ts, tc)
                if not rids:
                    continue
                g = summ[(summ.model == m) & (summ.train_set == ts) & (summ.train_cond == tc)
                         & (summ.test_cond == "orig")].set_index("test_set").macro_f1
                ind, cross, rdc = g.get(ts), g.get(OTHER[ts]), g.get("rdc")
                ind_bin = binary_in_domain(rids, ts)
                rows.append({"model": m, "train_set": ts, "train_cond": tc, "in_domain": ind,
                             "cross_3class": cross, "cross_dsl3m": g.get("dsl3m"), "rdc": rdc,
                             "in_domain_binary": ind_bin, "drop_3class": ind - cross, "drop_rdc": ind_bin - rdc})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table3_cross_dataset.csv", index=False, float_format="%.4f")
    body = " & & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{train {TS_NAME[ts]}}}" for ts in GRID["train_sets"]) + " \\\\\n"
    body += "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\n"
    body += ("Model & train cond. & in-dom. & $\\to$DSD-3 & $\\to$RDC & $\\Delta_3$ & $\\Delta_{\\mathrm{RDC}}$ "
             "& in-dom. & $\\to$DepSign & $\\to$RDC & $\\Delta_3$ & $\\Delta_{\\mathrm{RDC}}$ \\\\\n\\midrule\n")
    for m in t.model.unique():
        for i, tc in enumerate(CONDS):
            cells = []
            for ts in GRID["train_sets"]:
                r = t[(t.model == m) & (t.train_set == ts) & (t.train_cond == tc)]
                if not len(r):
                    cells += ["--"] * 5
                    continue
                r = r.iloc[0]
                cells += [f3(r.in_domain), f3(r.cross_3class), f3(r.rdc), f3(r.drop_3class), f3(r.drop_rdc)]
            if len(t[(t.model == m) & (t.train_cond == tc)]):
                body += (NAME[m] if i == 0 else "") + f" & {tc} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace[2pt]\n"
    write_tex("table3_cross_dataset", body,
              "Cross-dataset macro-F1 (unmasked test sets, mean over seeds). RDC is binary (P(moderate)+P(severe) vs.\\ P(none)). "
              "$\\Delta_3$: in-domain $-$ 3-class cross-dataset F1. $\\Delta_{\\mathrm{RDC}}$: binary-collapsed in-domain F1 $-$ RDC F1 "
              "(both binary, so the drop is like-for-like). Lower $\\Delta$ = better transfer.",
              "tab:cross", "ll" + "c" * 10)
    return t


# ---------------------------------------------------------------- Table 4: KRG with paired bootstrap CI
def table4(master):
    rows = []
    models = present(master, MODELS)
    for m in models:
        for tc in ["orig", "mask-topic"]:
            for tr in sorted(master[(master.model == m) & (master.train_cond == tc)].train_set.unique()):
                rids = seeds_of(master, m, tr, tc)
                for te in GRID["test_sets"]:
                    if not all(pred_path(r, te, c).exists() for r in rids for c in CONDS):
                        continue
                    krg, lo, hi = paired_diff(rids, te, "mask-random", "mask-topic")
                    drop_t = paired_diff(rids, te, "orig", "mask-topic")
                    drop_r = paired_diff(rids, te, "orig", "mask-random")
                    rows.append({"model": m, "train_set": tr, "train_cond": tc, "test_set": te,
                                 "n_seeds": len(rids), "KRG": krg, "KRG_lo": lo, "KRG_hi": hi,
                                 "drop_topic": drop_t[0], "drop_topic_lo": drop_t[1], "drop_topic_hi": drop_t[2],
                                 "drop_random": drop_r[0], "drop_random_lo": drop_r[1], "drop_random_hi": drop_r[2],
                                 "ci_excludes_0": lo > 0 or hi < 0})
                    print(f"  KRG {m:13s} {tr:7s} {tc:10s} -> {te:7s} {krg:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "table4_krg.csv", index=False, float_format="%.4f")

    def cell(r):
        star = "$^{*}$" if r.ci_excludes_0 else ""
        return f"\\shortstack{{{r.KRG:+.3f}{star}\\\\[-1pt]\\tiny[{r.KRG_lo:+.3f}, {r.KRG_hi:+.3f}]}}"

    head = " & ".join(f"{TS_NAME[a]}$\\to${TS_NAME[b]}" for a, b in SETTINGS)
    body = f"Model & train cond. & {head} & rand.\\ drop (in-dom.) \\\\\n\\midrule\n"
    for m in models:
        for tc in ["orig", "mask-topic"]:
            sub = t[(t.model == m) & (t.train_cond == tc)]
            if not len(sub):
                continue
            cells = []
            for a, b in SETTINGS:
                tr = "none" if m == "qwen-zs" else a  # zero-shot has no train set
                r = sub[(sub.train_set == tr) & (sub.test_set == b)]
                cells.append(cell(r.iloc[0]) if len(r) else "--")
            ind = sub[sub.train_set == sub.test_set]
            rd = "/".join(f"{x:+.3f}" for x in ind.sort_values("train_set").drop_random) if len(ind) else "--"
            label = NAME[m] if m != "qwen-fs" else NAME[m] + " (shots from train set)"
            body += f"{label} & {tc} & " + " & ".join(cells) + f" & {rd} \\\\\n"
    write_tex("table4_krg", body,
              "Keyword-reliance gap KRG $=$ F1(mask-random test) $-$ F1(mask-topic test), mean over seeds, with paired bootstrap 95\\% CI "
              "(1{,}000 resamples shared across seeds and conditions). $^{*}$: CI excludes 0. Positive = the model loses more from "
              "removing topic words than from removing the same number of random words. Last column: F1 drop from random masking alone "
              "(DepSign/DSD-3 in-domain). Qwen-ZS has no training set, so its two RDC cells are the same run.",
              "tab:krg", "ll" + "c" * 5)
    return t


# ---------------------------------------------------------------- McNemar + Holm
def mcnemar_family(master, a_cond, b_cond, family):
    rows = []
    for m in present(master, TRAINED):
        for ts in GRID["train_sets"]:
            ra, rb = seeds_of(master, m, ts, a_cond), seeds_of(master, m, ts, b_cond)
            if not ra or not rb:
                continue
            for te in GRID["test_sets"]:
                for tc in ["orig", "mask-topic"]:
                    ids, y, _ = load_pred(ra[0], te, "orig")
                    pa = majority_vote([load_pred(r, te, tc, ids)[2] for r in ra])
                    pb = majority_vote([load_pred(r, te, tc, ids)[2] for r in rb])
                    k = N_CLASSES[te]
                    res = mcnemar(y, pa, pb)
                    rows.append({"family": family, "model": m, "train_set": ts, "test_set": te, "test_cond": tc,
                                 "A": a_cond, "B": b_cond, "f1_A_vote": f1_point(y, pa, k),
                                 "f1_B_vote": f1_point(y, pb, k), "acc_A": float((pa == y).mean()),
                                 "acc_B": float((pb == y).mean()), **res})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    t["significant"] = t["p_holm"] < 0.05
    return t


# ---------------------------------------------------------------- Table 7: LLM prompting
def table7(master):
    rows = []
    for m in ["qwen-zs", "qwen-fs"]:
        for rid in sorted(master[master.model == m].run_id.unique()):
            stats = {(s["test_set"], s["test_cond"]): s for s in json.loads((RUNS / rid / "run.json").read_text())["stats"]}
            for te in GRID["test_sets"]:
                f = pred_path(rid, te, "orig")
                if not f.exists():
                    continue
                df = pd.read_parquet(f)
                k = N_CLASSES[te]
                probs = df[["p0", "p1", "p2"]].to_numpy()
                probs = collapse_binary(probs) if k == 2 else probs
                y, p = df.label.to_numpy().astype(np.int64), probs.argmax(1)
                ok = df.parsed.to_numpy()
                unp = stats.get((te, "orig"), {}).get("unparseable_rate", float(1 - ok.mean()))
                rows.append({"model": m, "run_id": rid, "shots_from": rid.split("__")[1], "test_set": te,
                             "macro_f1": f1_point(y, p, k), "unparseable_rate": unp,
                             "macro_f1_parsed_only": f1_point(y[ok], p[ok], k) if ok.any() else np.nan,
                             "pred_share_c0": float((p == 0).mean())})
    t = pd.DataFrame(rows)
    if not len(t):
        return t
    t.to_csv(TAB / "table7_llm_prompting.csv", index=False, float_format="%.4f")
    body = "Setting & " + " & ".join(TS_NAME[te] for te in GRID["test_sets"]) + " \\\\\n\\midrule\n"
    for (m, src), sub in t.groupby(["model", "shots_from"], sort=False):
        sub = sub.set_index("test_set")
        lab = "zero-shot" if m == "qwen-zs" else f"5-shot (shots: {TS_NAME[src]})"
        cells = []
        for te in GRID["test_sets"]:
            if te not in sub.index:
                cells.append("--")
                continue
            r = sub.loc[te]
            cells.append(f"\\shortstack{{{r.macro_f1:.3f} ({r.macro_f1_parsed_only:.3f})\\\\[-1pt]\\tiny unp.\\ {100 * r.unparseable_rate:.1f}\\%}}")
        body += lab + " & " + " & ".join(cells) + " \\\\\n"
    write_tex("table7_llm_prompting", body,
              "Qwen2.5-1.5B-Instruct prompting, macro-F1 on unmasked test sets (unparseable outputs counted wrong); "
              "in parentheses: F1 on parseable outputs only; below: unparseable-output rate. Greedy decoding, one run (seed 13).",
              "tab:llm", "l" + "c" * len(GRID["test_sets"]))
    return t


# ---------------------------------------------------------------- figures
def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=INK2)
    ax.grid(axis="y", color=GRIDC, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def fig_heatmaps(master, summ):
    cmap = ListedColormap(BLUE_RAMP)
    for ts in GRID["train_sets"]:
        labels, mat = [], []
        for m in present(master, MODELS):
            for tc in ["orig", "mask-topic"]:
                g = summ[(summ.model == m) & (summ.train_set == (ts if m != "qwen-zs" else "none"))
                         & (summ.train_cond == tc) & (summ.test_cond == "orig")].set_index("test_set").macro_f1
                if not len(g):
                    continue
                labels.append(f"{NAME[m]} · {tc}" if m in TRAINED else NAME[m])
                mat.append([g.get(te, np.nan) for te in GRID["test_sets"]])
        mat = np.array(mat)
        fig, ax = plt.subplots(figsize=(4.6, 0.32 * len(labels) + 1.0))
        im = ax.imshow(mat, cmap=cmap, vmin=0.25, vmax=0.85, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    v = (mat[i, j] - 0.25) / 0.6
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="#ffffff" if v > 0.55 else INK)
        ax.set_xticks(range(len(GRID["test_sets"])), [TS_NAME[te] + (" (bin.)" if te == "rdc" else "")
                                                      for te in GRID["test_sets"]])
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(length=0, labelsize=7, colors=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, mat.shape[1]), minor=True)
        ax.set_yticks(np.arange(-.5, mat.shape[0]), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"Macro-F1, trained on {TS_NAME[ts]} (unmasked test)", fontsize=9, color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, length=0, colors=INK2)
        save(fig, f"heatmap_{ts}")


def fig_krg(krg):
    k = krg[krg.train_cond == "orig"]
    models = [m for m in MODELS if m in set(k.model)]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for i, (a, b) in enumerate(SETTINGS):
            r = k[(k.model == m) & (k.train_set == ("none" if m == "qwen-zs" else a)) & (k.test_set == b)]
            if m == "qwen-zs" and a == "dsl3" and b == "rdc":
                r = r.iloc[0:0]  # zero-shot shown once per test set
            if len(r):
                r = r.iloc[0]
                xs.append(i - 0.4 + w * (j + .5)); ys.append(r.KRG); lo.append(r.KRG - r.KRG_lo); hi.append(r.KRG_hi - r.KRG)
        ax.bar(xs, ys, width=w * 0.85, color=MODEL_COLOR[m], label=NAME[m], zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5, zorder=4)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(SETTINGS)), [f"{TS_NAME[a]} → {TS_NAME[b]}" for a, b in SETTINGS])
    ax.set_ylabel("KRG (macro-F1)", fontsize=8, color=INK2)
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=min(4, len(models)), loc="upper left", bbox_to_anchor=(0, 1.22))
    save(fig, "krg_bars")


def fig_transfer(t3):
    """3-class transfer drop (in-domain F1 - F1 on the other 3-class dataset), per training condition."""
    models = [m for m in TRAINED if m in set(t3.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    w = 0.8 / len(CONDS)
    for ax, ts in zip(axes, GRID["train_sets"]):
        for j, tc in enumerate(CONDS):
            sub = t3[(t3.train_set == ts) & (t3.train_cond == tc)].set_index("model")
            xs = [i - 0.4 + w * (j + .5) for i, m in enumerate(models) if m in sub.index]
            ys = [sub.loc[m, "drop_3class"] for m in models if m in sub.index]
            ax.bar(xs, ys, width=w * 0.85, color=COND_COLOR[tc], label=tc, zorder=3)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=2)
        ax.set_xticks(range(len(models)), [NAME[m] for m in models], rotation=25, ha="right")
        ax.set_title(f"{TS_NAME[ts]} → {TS_NAME[OTHER[ts]]}", fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("Transfer drop (macro-F1)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.3))
    save(fig, "transfer_drop")


def fig_rdc(t3):
    """RDC binary macro-F1 per training condition (dot plot: no zero baseline needed)."""
    models = [m for m in TRAINED if m in set(t3.model)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    off = np.linspace(-0.27, 0.27, len(CONDS))
    for ax, ts in zip(axes, GRID["train_sets"]):
        for j, tc in enumerate(CONDS):
            sub = t3[(t3.train_set == ts) & (t3.train_cond == tc)].set_index("model")
            xs = [i + off[j] for i, m in enumerate(models) if m in sub.index]
            ys = [sub.loc[m, "rdc"] for m in models if m in sub.index]
            ax.scatter(xs, ys, s=30, color=COND_COLOR[tc], edgecolor="#fcfcfb", linewidth=1.2, label=tc, zorder=3)
        ax.set_xticks(range(len(models)), [NAME[m] for m in models], rotation=25, ha="right")
        ax.set_title(f"Train {TS_NAME[ts]} → RDC (binary)", fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("RDC macro-F1", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.3))
    save(fig, "rdc_transfer")


# ---------------------------------------------------------------- main
def main():
    master = pd.read_csv(MASTER)
    summ = seed_summary(master)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("table 2 ..."); t2 = table2(master, summ)
    print("table 3 ..."); t3 = table3(master, summ)
    print("table 4 (paired bootstrap) ..."); t4 = table4(master)
    print("McNemar + Holm ...")
    mc = pd.concat([mcnemar_family(master, "orig", "mask-topic", "orig_vs_mask-topic"),
                    mcnemar_family(master, "mask-random", "mask-topic", "mask-random_vs_mask-topic")])
    mc.to_csv(TAB / "mcnemar_holm.csv", index=False, float_format="%.5g")
    print("table 7 ..."); t7 = table7(master)
    print("figures ...")
    fig_heatmaps(master, summ); fig_krg(t4); fig_transfer(t3); fig_rdc(t3)
    key = {"n_master_rows": len(master), "models": sorted(master.model.unique()),
           "mcnemar": mc.groupby("family").agg(n=("p", "size"), n_sig=("significant", "sum")).reset_index().to_dict("records")}
    (TAB / "key_numbers.json").write_text(json.dumps(key, indent=1, default=str))
    print(f"done: {len(t2)} table-2 rows, {len(t3)} table-3 rows, {len(t4)} KRG rows, {len(mc)} McNemar tests, "
          f"{len(t7)} LLM rows")


if __name__ == "__main__":
    main()
