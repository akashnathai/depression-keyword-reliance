"""Phase 7: figures and tables in the form used in the paper.

  python scripts/phase7_diagram_figs.py

Reads only results/ and data/processed/dataset_stats.csv; writes results/diagram_Figs/figures/*.pdf and
results/diagram_Figs/tables/*.tex.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from phase5_report import AXIS, GRIDC, INK, INK2, SLOTS, style  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
TAB, FIG = ROOT / "results/tables", ROOT / "results/figures"
PT, PF = ROOT / "results/diagram_Figs/tables", ROOT / "results/diagram_Figs/figures"
PT.mkdir(parents=True, exist_ok=True)
PF.mkdir(parents=True, exist_ok=True)

DS = {"depsign": "DepSign", "dsl3": "DSD-3", "dsl3m": "DSD-3m", "rdc": "RDC"}
MODELS = ["tfidf-lr", "bert", "roberta", "mentalbert", "mentalroberta", "qwen-qlora", "qwen-zs", "qwen-fs"]
TRAINED = MODELS[:6]
NAME = {"tfidf-lr": "TF-IDF+LR", "bert": "BERT", "roberta": "RoBERTa", "mentalbert": "MentalBERT",
        "mentalroberta": "MentalRoBERTa", "qwen-qlora": "Qwen-QLoRA", "qwen-zs": "Qwen-ZS", "qwen-fs": "Qwen-FS"}
MCOL = {m: SLOTS[i] for i, m in enumerate(MODELS)}
COND3 = ["orig", "mask-topic", "mask-random"]
CCOL = {"orig": SLOTS[0], "mask-topic": SLOTS[1], "mask-random": SLOTS[3]}
EXPL = {"ig": "IG", "shap": "SHAP", "lime": "LIME", "random": "Random"}
TAG = {"orig": "orig-trained", "mask-topic": "mask-topic-trained"}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(PF / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(PF / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def tt(c):
    return "\\texttt{" + c + "}"


# ------------------------------------------------------------------ Fig. 1 keyword prevalence
def fig1():
    k = pd.read_csv(TAB / "keyword_prevalence.csv")
    k = k[k.condition.isin(["mask-topic", "mask-topic+sui"])]
    order = ["depsign", "dsl3", "rdc"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True, gridspec_kw={"width_ratios": [3, 3, 2]})
    lab = {"mask-topic": "topic words", "mask-topic+sui": "topic or suicide words"}
    col = {"mask-topic": SLOTS[1], "mask-topic+sui": SLOTS[6]}
    for ax, mp in zip(axes, order):
        sub = k[k.mapping == mp]
        classes = list(dict.fromkeys(sub.sort_values("label")["class"]))
        for j, c in enumerate(["mask-topic", "mask-topic+sui"]):
            s = sub[sub.condition == c].sort_values("label")
            xs = np.arange(len(classes)) + (j - 0.5) * 0.36
            ax.bar(xs, s.pct_posts_masked, width=0.34, color=col[c], label=lab[c], zorder=3)
        ax.set_xticks(range(len(classes)), [c.replace("not ", "not\n").replace("+", "+\n") for c in classes],
                      fontsize=7)
        ax.set_title(DS[mp] + (" (binary)" if mp == "rdc" else ""), fontsize=8.5, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("% posts with ≥1 match", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    save(fig, "fig1_prevalence")


# ------------------------------------------------------------------ Fig. 2 KRG bars
def fig2():
    k = pd.read_csv(TAB / "table4_krg.csv")
    k = k[k.train_cond == "orig"]
    settings = [("depsign", "depsign"), ("depsign", "rdc"), ("dsl3", "dsl3"), ("dsl3", "rdc")]
    models = [m for m in MODELS if m in set(k.model)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.84 / len(models)
    for j, m in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for i, (a, b) in enumerate(settings):
            tr = "none" if m == "qwen-zs" else a
            if m == "qwen-zs" and (a, b) == ("dsl3", "rdc"):
                continue  # zero-shot has no training set; its RDC value is shown once
            r = k[(k.model == m) & (k.train_set == tr) & (k.test_set == b)]
            if len(r):
                r = r.iloc[0]
                xs.append(i - 0.42 + w * (j + .5)); ys.append(r.KRG)
                lo.append(r.KRG - r.KRG_lo); hi.append(r.KRG_hi - r.KRG)
        ax.bar(xs, ys, width=w * 0.86, color=MCOL[m], label=NAME[m], zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.2, zorder=4)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=2)
    ax.set_xticks(range(len(settings)), [f"{DS[a]} → {DS[b]}" + (" (binary)" if b == "rdc" else "") for a, b in settings],
                  fontsize=7.5)
    ax.set_ylabel("KRG (macro-F1)", fontsize=8, color=INK2)
    style(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.27))
    save(fig, "fig2_krg")


# ------------------------------------------------------------------ Fig. 3 transfer dot plot
def fig3():
    t3 = pd.read_csv(TAB / "table3_cross_dataset.csv")
    d = pd.read_csv(TAB / "train_cond_diff.csv")
    panels = [("depsign", "dsl3", "cross_3class"), ("dsl3", "depsign", "cross_3class"),
              ("depsign", "rdc", "rdc"), ("dsl3", "rdc", "rdc")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    axes = axes.ravel()
    off = {"orig": -0.2, "mask-topic": 0.0, "mask-random": 0.2}
    for ax, (tr, te, col) in zip(axes, panels):
        for c in COND3:
            sub = t3[(t3.train_set == tr) & (t3.train_cond == c)].set_index("model")
            for i, m in enumerate(TRAINED):
                if m not in sub.index:
                    continue
                y = sub.loc[m, col]
                filled = True
                if c != "orig":
                    r = d[(d.model == m) & (d.train_set == tr) & (d.test_set == te) & (d.A == c) & (d.B == "orig")]
                    filled = bool(r.ci_excludes_0.iloc[0]) if len(r) else False
                ax.scatter(i + off[c], y, s=30, marker="o", zorder=3, linewidths=1.2,
                           facecolor=CCOL[c] if filled else "white", edgecolor=CCOL[c])
        ax.set_xticks(range(len(TRAINED)), [NAME[m] for m in TRAINED], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{DS[tr]} → {DS[te]}" + (" (binary)" if te == "rdc" else ""), fontsize=8, color=INK, loc="left")
        style(ax)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[::2]:
        ax.set_ylabel("macro-F1", fontsize=7.5, color=INK2)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=CCOL[c],
                          markeredgecolor=CCOL[c], label=f"{c}-trained") for c in COND3]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="white",
                              markeredgecolor=INK2, label="hollow: CI of difference to orig includes 0"))
    fig.legend(handles=handles, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0, 1, 0.96))
    save(fig, "fig3_transfer")


# ------------------------------------------------------------------ Fig. 4 attribution ratio
def fig4():
    a = pd.read_csv(TAB / "attribution_ratio.csv")
    groups = [("topic", SLOTS[1]), ("suicide", SLOTS[7]), ("symptom", SLOTS[2])]
    rows = [(m, t) for t in ("orig", "mask-topic") for m in ("ig", "shap", "lime")]
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    for gi, (g, colr) in enumerate(groups):
        for ri, (m, t) in enumerate(rows):
            r = a[(a.model == t) & (a.method == m) & (a.group == g)].iloc[0]
            y = ri + (gi - 1) * 0.22
            ax.errorbar(r.ratio, y, xerr=[[r.ratio - r.ci_lo], [r.ci_hi - r.ratio]], fmt="o", ms=4,
                        color=colr, ecolor=colr, elinewidth=1, capsize=1.5, label=f"{g} words" if ri == 0 else None)
    ax.axvline(1, color=INK2, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5×", "1×", "2×", "4×", "8×"])
    ax.set_yticks(range(len(rows)), [f"{EXPL[m]} · {TAG[t]}" for m, t in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("attribution ratio (share of positive attribution ÷ share of words)", fontsize=7.5, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=7, colors=INK2)
    ax.grid(axis="x", color=GRIDC, linewidth=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.18))
    save(fig, "fig4_attribution_ratio")


# ------------------------------------------------------------------ Fig. 5 faithfulness curves
def fig5():
    f = pd.read_csv(TAB / "table5_faithfulness.csv")
    ks = [1, 5, 10, 20, 50]
    col = {"ig": SLOTS[0], "shap": SLOTS[1], "lime": SLOTS[2], "random": "#898781"}
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2))
    axes = axes.ravel()
    for ax, (t, kind) in zip(axes, [(t, k) for k in ("comp", "suff") for t in ("orig", "mask-topic")]):
        sub = f[f.model == t].set_index("method")
        for m in ("ig", "shap", "lime", "random"):
            ax.plot(ks, [sub.loc[m, f"{kind}_{k}"] for k in ks], color=col[m], linewidth=1.6, marker="o", ms=3,
                    linestyle="--" if m == "random" else "-", label=EXPL[m])
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_title(("Comprehensiveness" if kind == "comp" else "Sufficiency") + f", {TAG[t]}", fontsize=8,
                     color=INK, loc="left")
        style(ax)
        ax.tick_params(labelsize=7)
    for ax in axes[2:]:
        ax.set_xlabel("% of words removed (compr.) or kept (suff.)", fontsize=7, color=INK2)
    for ax in axes[::2]:
        ax.set_ylabel("Δ p(predicted class)", fontsize=7.5, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0, 1, 0.96))
    save(fig, "fig5_faithfulness")


# ------------------------------------------------------------------ tables
def table_wrap(body, caption, label, colspec, size="\\scriptsize"):
    return ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\\label{" + label + "}\n" + size +
            "\n\\setlength{\\tabcolsep}{3pt}\n\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}{" + colspec +
            "}\n\\toprule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")


def tab1():
    s = pd.read_csv(ROOT / "data/processed/dataset_stats.csv")
    order = [("depsign", s_) for s_ in ("train", "dev", "test")] + [("dsl3", s_) for s_ in ("train", "dev", "test")] + \
            [("dsl3m", s_) for s_ in ("train", "dev", "test")] + [("rdc", "test")]
    body = "Dataset & Split & $N$ & none & moderate & severe & words \\\\\n\\midrule\n"
    prev = None
    for mp, sp in order:
        r = s[(s.mapping == mp) & (s.split == sp)].iloc[0]
        c2 = "--" if pd.isna(r.c2) else f"{int(r.c2):,}"
        name = DS[mp] + (" (binary)" if mp == "rdc" else "") if mp != prev else ""
        if prev and mp != prev:
            body += "\\addlinespace[2pt]\n"
        body += f"{name} & {sp} & {int(r.n):,} & {int(r.c0):,} & {int(r.c1):,} & {c2} & {r.mean_words:.0f} \\\\\n"
        prev = mp
    cap = ("Posts per split and class after cleaning and deduplication, and mean length in words. "
           "DSD-3 maps minimum/mild+moderate/severe and DSD-3m maps minimum+mild/moderate/severe onto the three "
           "classes (same texts). RDC is test-only and binary (not depressed/depressed).")
    (PT / "tab1_data.tex").write_text(table_wrap(body, cap, "tab:data", "llrrrrr"), encoding="utf-8")


def tab2():
    t2 = pd.read_csv(TAB / "table2_in_domain.csv")
    t7 = pd.read_csv(TAB / "table7_llm_prompting.csv")
    hdr = " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DS[ts]}}}" for ts in ("depsign", "dsl3")) + " \\\\\n"
    hdr += "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nModel & " + " & ".join(tt(c) for c in COND3 * 2) + " \\\\\n\\midrule\n"
    body = hdr
    for m in TRAINED:
        cells = []
        for ts in ("depsign", "dsl3"):
            sub = t2[(t2.model == m) & (t2.train_set == ts)].set_index("train_cond")
            best = sub.loc[[c for c in COND3 if c in sub.index], "macro_f1"].max()
            for c in COND3:
                if c not in sub.index:
                    cells.append("--")
                    continue
                r = sub.loc[c]
                v = f"{r.macro_f1:.3f}"
                v = f"\\textbf{{{v}}}" if np.isclose(r.macro_f1, best) else v
                sd = ("" if r.sd == 0 else f"\\,$\\pm$\\,{r.sd:.3f}") if r.n_seeds > 1 else "$^{\\dagger}$"
                cells.append(v + sd)
        body += NAME[m] + " & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    for m, src_dep, src_dsd in [("qwen-zs", "none", "none"), ("qwen-fs", "depsign", "dsl3")]:
        d = t7[(t7.model == m) & (t7.shots_from == src_dep) & (t7.test_set == "depsign")].iloc[0].macro_f1
        s = t7[(t7.model == m) & (t7.shots_from == src_dsd) & (t7.test_set == "dsl3")].iloc[0].macro_f1
        body += f"{NAME[m]} & {d:.3f}$^{{\\dagger}}$ & -- & -- & {s:.3f}$^{{\\dagger}}$ & -- & -- \\\\\n"
    cap = ("In-domain macro-F1 on the \\texttt{orig} test set by training condition (mean $\\pm$ s.d.\\ over three "
           "seeds; TF-IDF+LR is deterministic). Bold: best training condition per model and dataset. $^{\\dagger}$Single run. Qwen-ZS and Qwen-FS "
           "are prompted (no training); Qwen-FS uses five examples from the training split of the same dataset. "
           "Bootstrap CIs and \\texttt{mask-topic+sui} results are in the supplement.")
    (PT / "tab2_indomain.tex").write_text(table_wrap(body, cap, "tab:indomain", "lcccccc"), encoding="utf-8")


def tab3():
    k = pd.read_csv(TAB / "table4_krg.csv")
    settings = [("depsign", "depsign"), ("depsign", "rdc"), ("dsl3", "dsl3"), ("dsl3", "rdc")]

    def cell(r):
        star = "$^{*}$" if r.ci_excludes_0 else ""
        return f"{r.KRG:+.3f}{star} \\tiny[{r.KRG_lo:+.3f}, {r.KRG_hi:+.3f}]"

    head = " & ".join(f"{DS[a]}$\\to${DS[b]}" for a, b in settings)
    body = f"Model & {head} \\\\\n\\midrule\n\\multicolumn{{5}}{{l}}{{\\textit{{orig-trained}}}} \\\\\n"
    for m in MODELS:
        sub = k[(k.model == m) & (k.train_cond == "orig")]
        cells = []
        for a, b in settings:
            tr = "none" if m == "qwen-zs" else a
            r = sub[(sub.train_set == tr) & (sub.test_set == b)]
            cells.append(cell(r.iloc[0]) if len(r) else "--")
        dag = "$^{\\dagger}$" if m.startswith("qwen") else ""
        body += NAME[m] + dag + " & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n\\multicolumn{5}{l}{\\textit{mask-topic-trained}} \\\\\n"
    for m in TRAINED:
        sub = k[(k.model == m) & (k.train_cond == "mask-topic")]
        cells = [cell(sub[(sub.train_set == a) & (sub.test_set == b)].iloc[0]) for a, b in settings]
        dag = "$^{\\dagger}$" if m.startswith("qwen") else ""
        body += NAME[m] + dag + " & " + " & ".join(cells) + " \\\\\n"
    cap = ("Keyword-reliance gap, KRG $=$ macro-F1(\\texttt{mask-random} test) $-$ macro-F1(\\texttt{mask-topic} test), "
           "mean over seeds with paired bootstrap 95\\% CI. $^{*}$CI excludes 0. RDC is binary. $^{\\dagger}$Single run; "
           "Qwen-ZS has no training set, so both RDC columns show the same run.")
    (PT / "tab3_krg.tex").write_text(table_wrap(body, cap, "tab:krg", "lcccc"), encoding="utf-8")


def tab4():
    f = pd.read_csv(TAB / "table5_faithfulness.csv")
    a = pd.read_csv(TAB / "attribution_ratio.csv")

    def ci(v, lo, hi, d=3):
        return f"{v:.{d}f} \\tiny[{lo:.{d}f}, {hi:.{d}f}]"

    body = ("Model & Explainer & Compr.\\ AOPC $\\uparrow$ & Suff.\\ AOPC $\\downarrow$ & Topic ratio & Symptom ratio \\\\\n"
            "\\midrule\n")
    for t in ("orig", "mask-topic"):
        for i, m in enumerate(("ig", "shap", "lime", "random")):
            r = f[(f.model == t) & (f.method == m)].iloc[0]
            cells = [ci(r.aopc_comp, r.aopc_comp_lo, r.aopc_comp_hi), ci(r.aopc_suff, r.aopc_suff_lo, r.aopc_suff_hi)]
            if m == "random":
                cells += ["--", "--"]
            else:
                for g in ("topic", "symptom"):
                    q = a[(a.model == t) & (a.method == m) & (a.group == g)].iloc[0]
                    cells.append(ci(q.ratio, q.ci_lo, q.ci_hi, 2))
            body += (TAG[t] if i == 0 else "") + f" & {EXPL[m]} & " + " & ".join(cells) + " \\\\\n"
        if t == "orig":
            body += "\\addlinespace[2pt]\n"
    cap = ("Faithfulness and attribution ratios for MentalRoBERTa on 300 DepSign test posts (100 per class), "
           "explaining the predicted class. AOPC averages the probability drop over $k\\in\\{1,5,10,20,50\\}\\%$ of words "
           "(comprehensiveness: top words deleted, higher is better; sufficiency: only top words kept, lower is better). "
           "Random: mean of five random rankings. Ratio: share of positive attribution $\\div$ share of words "
           "(1 = proportional). Bootstrap 95\\% CIs over posts.")
    (PT / "tab4_xai.tex").write_text(table_wrap(body, cap, "tab:xai", "llcccc"), encoding="utf-8")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    tab1(); tab2(); tab3(); tab4()
    print("written:", sorted(p.name for p in PF.glob("*.pdf")), sorted(p.name for p in PT.glob("*.tex")))
