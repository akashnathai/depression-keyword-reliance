# Reddit Depression-Severity Classifiers with keyword resiliance

Code and results for our study of shortcut learning in depression-severity classification on
Reddit. We mask words that name depression or its treatment and compare the effect with masking
the same number of random words. The difference is the **keyword-reliance gap (KRG)**. We evaluate
a TF-IDF baseline, general and mental-health encoders, and a small instruction-tuned LLM, and use
attribution methods to see which words the models rely on.

**Authors:** Akash Nath, Pragyat Jyoti Baruah, Arnab Paul, Tirthanka Borah
(Department of Computer Science and Engineering, Assam University, Silchar, India).

Method details are in the accompanying paper.

> **Data and ethics.** All datasets are public and used under their own terms. They are not
> redistributed here, and no post text appears in this repository. This work is about how
> classifiers behave in research settings; it does not support diagnosis or any clinical use.

---

## Main findings

- Models trained on DepSign rely clearly on topic words. On DSD the reliance is close to zero.
- Mental-health pretraining reduced this reliance for only one of the two encoder pairs.
- Relative to how often they occur, topic words receive far more attribution than symptom words.
- Training on masked text lowers the reliance but does not reliably improve cross-dataset transfer.

All numbers are produced from `results/` by the scripts below.

---

## 1. Environment

- Python 3.11, one CUDA GPU (developed on an RTX 4060 Laptop, 8 GB) and 16 GB RAM.
- Install:
  ```bash
  conda create -n depr python=3.11 -y && conda activate depr
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  ```
- Hugging Face: `hf auth login`, and accept the terms of the gated models
  `mental/mental-bert-base-uncased` and `mental/mental-roberta-base`. Set `HF_HOME` to choose
  where models are cached.
- On Windows the scripts cap PyTorch VRAM (`vram_fraction` in `configs/qwen.yaml`). Without the
  cap, the WDDM driver silently moves GPU memory into system RAM and runs become very slow.

## 2. Data (download yourself; not redistributed)

| key | source | put files in |
|---|---|---|
| depsign | DepSign-LT-EDI@ACL 2022; copy with test labels from `rafalposwiata/depression-detection-lt-edi-2022` (`data/original_dataset`) | `data/raw/depsign/` (train/dev/test `.tsv`) |
| dsd | Naseem et al. (WWW 2022), `usmaann/Depression_Severity_Dataset` | `data/raw/dsd/Reddit_depression_dataset.csv` |
| rdc | HF `mrjunos/depression-reddit-cleaned` (CC BY 4.0) | `data/raw/rdc/depression_reddit_cleaned_ds.csv` |
| primate | Gupta et al. (CLPsych 2022); **signed agreement required** | `data/raw/primate/` (optional) |
| redsm5 | HF `irlab-udc/redsm5`; gated | `data/raw/redsm5/` (optional) |

## 3. Reproduce (in order)

Every step is config-driven, seeded and resumable; a run whose prediction file exists is skipped.

| step | command | output | time on RTX 4060 |
|---|---|---|---|
| 1 data | `python scripts/phase1_data.py` | `data/processed/`, dedup report, Table 1 | ~5 min |
| 1b masking | `python scripts/phase1b_mask.py` | `data/masked/<condition>/`, keyword prevalence | ~2 min |
| 2 TF-IDF | `python scripts/phase2_tfidf.py` | 24 runs → `results/preds/` | ~20 min (CPU) |
| 3 encoders | `python scripts/phase3_encoders.py` | 96 runs (4 models × 2 train sets × 4 conditions × 3 seeds) | ~10.5 h |
| 4a prompting | `python scripts/phase4_llm.py --prompt` | Qwen zero-shot and 5-shot | ~3 h |
| 4b QLoRA | `python scripts/phase4_llm.py --qlora --seeds 13` | 4 runs (single seed) | see `results/runs/*/run.json` |
| 5 report | `python scripts/phase5_report.py` and `python scripts/phase5_extra.py` | tables, McNemar + Holm, figures | ~5 min (CPU) |
| 6 XAI | `python scripts/phase6_xai.py --explain` then `--summarise` | faithfulness and attribution tables, XAI figures | ~45 min |
| 7 diagram_Figs | `python scripts/phase7_diagram_figs.py` | `results/diagram_Figs/{figures,tables}` | ~1 min (CPU) |
| check | `python src/eval/aggregate.py --full` | rebuilds `results/master_results.csv` from all predictions | ~10 min |

`python scripts/progress.py` prints the progress and estimated time left of the encoder grid.

## 4. Layout

```
configs/          grid.yaml, tfidf_lr.yaml, encoders.yaml, qwen.yaml
lexicons/         topic.txt, suicide.txt, symptom.txt (+ README with sources)
src/data/         clean.py, dedup.py, labels.py, lexicon.py, mask.py
src/models/       tfidf_lr.py, train_encoder.py (also QLoRA), llm_prompt.py
src/eval/         metrics.py, stats.py (McNemar, Holm, majority vote), evaluate.py, aggregate.py
src/xai/          common.py, ig.py, shap_partition.py, lime_text.py, faithfulness.py, attribution_mass.py
scripts/          phase entry points (see table above)
results/
  master_results.csv   one row per (run, test set, test-time condition)
  tables/              result tables (CSV + LaTeX)
  figures/             figures (PDF + PNG)
  diagram_Figs/        figures and tables as used in the paper
  runs/<run_id>/run.json   training history, timing, peak VRAM per run
  data_reports/        deduplication report, dataset statistics, lexicon prevalence (counts only)
```

Run IDs follow `<model>__<train set>__<train condition>__s<seed>`.

## 5. Key definitions

- **Masking conditions:** `orig`; `mask-topic` (topic-lexicon matches → `[KW]`); `mask-topic+sui`
  (also suicide terms); `mask-random` (the same number of `[KW]` per post, placed on random
  non-lexicon words). Train-time and test-time conditions are independent.
- **Keyword-reliance gap:** KRG = F1(mask-random test) − F1(mask-topic test), with a paired
  bootstrap CI (1,000 resamples shared across seeds).
- **RDC collapse:** RDC is binary. P(depressed) = P(moderate) + P(severe).

**Limitations:** RDC is lemmatised and binary, so it is used only as a stress test. Labels are
crowd- or self-reported, the data are English Reddit posts only, and QLoRA and prompting use a
single seed.
