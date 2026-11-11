"""Phase 1b: create masked variants of every processed split and the prevalence table and figure.

Outputs
  data/masked/<condition>/<mapping>_<split>.parquet   (adds n_spans, n_words_masked, n_words)
  results/tables/keyword_prevalence.csv / .tex        per mapping x class x condition
  results/figures/keyword_prevalence.pdf / .png
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labels import CLASS_NAMES  # noqa: E402
from src.data.mask import CONDITIONS, apply_condition, words_in  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
PROC, MASKED = Path("data/processed"), Path("data/masked")
TAB, FIG = Path("results/tables"), Path("results/figures")


def main():
    rows = []
    for f in sorted(PROC.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["n_words"] = df["text"].map(words_in)
        for cond in CONDITIONS:
            out = df.copy()
            res = [apply_condition(t, cond, i) for t, i in zip(df["text"], df["id"])]
            out["text"] = [r[0] for r in res]
            out["n_spans"] = [r[1] for r in res]
            out["n_words_masked"] = [r[2] for r in res]
            (MASKED / cond).mkdir(parents=True, exist_ok=True)
            out.to_parquet(MASKED / cond / f.name, index=False)
            rows.append(out.assign(condition=cond)[["mapping", "split", "label", "condition",
                                                     "n_spans", "n_words_masked", "n_words"]])
        print("masked", f.name)
    all_rows = pd.concat(rows)
    all_rows.to_parquet(MASKED / "mask_counts.parquet", index=False)

    # prevalence (all splits pooled; dsl3m shares texts with dsl3, so it is omitted)
    prev = []
    for (mp, lab, cond), g in all_rows[all_rows["mapping"] != "dsl3m"].groupby(["mapping", "label", "condition"]):
        prev.append({"mapping": mp, "label": int(lab), "class": CLASS_NAMES[mp][int(lab)],
                     "condition": cond, "n_posts": len(g),
                     "pct_posts_masked": 100 * (g["n_spans"] > 0).mean(),
                     "pct_tokens_masked": 100 * g["n_words_masked"].sum() / g["n_words"].sum()})
    prev = pd.DataFrame(prev)
    TAB.mkdir(parents=True, exist_ok=True)
    prev.to_csv(TAB / "keyword_prevalence.csv", index=False)
    write_latex(prev)
    plot(prev)
    wide = prev.pivot_table(index=["mapping", "label", "class", "n_posts"], columns="condition",
                            values=["pct_posts_masked", "pct_tokens_masked"]).round(2)
    print(wide.to_string())


def write_latex(prev):
    p = prev.pivot_table(index=["mapping", "label", "class", "n_posts"], columns="condition",
                         values=["pct_posts_masked", "pct_tokens_masked"]).reset_index()
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Keyword prevalence by class: \% of posts with at least one masked span and"
             r" \% of word tokens masked (all splits pooled). mask-random masks the same number of"
             r" tokens per post as mask-topic.}", r"\label{tab:prevalence}",
             r"\begin{tabular}{llrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{mask-topic} & \multicolumn{2}{c}{mask-topic+sui} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Data & Class & $N$ & posts \% & tokens \% & posts \% & tokens \% \\", r"\midrule"]
    for _, r in p.iterrows():
        lines.append(f"{r[('mapping', '')]} & {r[('class', '')]} & {int(r[('n_posts', '')]):,} & "
                     f"{r[('pct_posts_masked', 'mask-topic')]:.1f} & {r[('pct_tokens_masked', 'mask-topic')]:.2f} & "
                     f"{r[('pct_posts_masked', 'mask-topic+sui')]:.1f} & {r[('pct_tokens_masked', 'mask-topic+sui')]:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / "keyword_prevalence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(prev):
    colors = {"mask-topic": "#2a78d6", "mask-topic+sui": "#eb6834"}
    maps = ["depsign", "dsl3", "rdc"]
    titles = {"depsign": "DepSign", "dsl3": "DSD (dsl3)", "rdc": "RDC (binary)"}
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), sharey=True,
                             gridspec_kw={"width_ratios": [3, 3, 2]})
    for ax, mp in zip(axes, maps):
        sub = prev[prev["mapping"] == mp]
        classes = CLASS_NAMES[mp]
        w = 0.36
        for j, cond in enumerate(colors):
            vals = [sub[(sub["label"] == c) & (sub["condition"] == cond)]["pct_posts_masked"].iloc[0]
                    for c in range(len(classes))]
            xs = [c + (j - 0.5) * (w + 0.03) for c in range(len(classes))]
            ax.bar(xs, vals, width=w, color=colors[cond], label=cond, zorder=3)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c.replace("+", "+\n") for c in classes], fontsize=8)
        ax.set_title(titles[mp], fontsize=9, loc="left")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9a998f")
        ax.tick_params(axis="y", length=0, labelsize=8, colors="#55544f")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% posts with ≥1 masked span", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "keyword_prevalence.pdf")
    fig.savefig(FIG / "keyword_prevalence.png", dpi=200)


if __name__ == "__main__":
    main()
