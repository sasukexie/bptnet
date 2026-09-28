# BPTNet — Bidirectional Phase-aware Transformer Network for Micro-Expression Recognition

Reference implementation and released artefacts for the manuscript
*"Bidirectional Phase-aware Transformer Network for Micro-Expression Recognition"*.

This repository contains (a) the training / evaluation pipeline used for every number in the paper,
(b) the scripts that regenerate the tables from the raw per-fold results, and (c) the released
per-fold split definitions, per-sample lists and per-fold confusion matrices
(see [`release/`](#released-artefacts)).

> We do **not** redistribute the original benchmark images or optical flow. All four benchmarks are
> publicly available from their original providers under their own terms; see
> [Data](#1-data) for the expected directory layout and the access conditions.

---

## 0. Quick start

```bash
conda create -n bptnet python=3.10 -y && conda activate bptnet
pip install -r requirements.txt

python run_main.py                                # full protocol, all four single-database settings
```

The benchmark data are read from `data.data_dir` in `src/config/common.yml` (default `dataset`), which
is resolved **relative to the directory that contains this repository** — i.e. the expected location is
`../dataset`. Put your copy of the benchmarks there, or edit `data_dir` in `common.yml`.
`MER_RESULTS_DIR` (optional environment variable) can redirect the result directory.

A short smoke run (one dataset, one seed, one input representation):

```bash
BPTNET_DATASETS=casme2_3c BPTNET_SEEDS=1024 BPTNET_FRAME_TYPES=flow python run_main.py
```

---

## 1. Data

The pipeline expects a *rich* layout (one directory per sample, RGB frames **and** both optical-flow
fields pre-computed). Directory names are the benchmark *sources*:

```
<data-root>/
  casme2_7c/rgb/<subject>/<sample>/*.png          # CASME II frames
  casme2_7c/flow/<subject>/<sample>/*.npy         # onset -> apex dense flow
  casme2_7c/flow_ao/<subject>/<sample>/*.npy      # apex -> offset dense flow
  samm_8c/{rgb,flow,flow_ao}/<subject>/<sample>/  # SAMM
  smic_hs_3c/{rgb,flow,flow_ao}/<subject>/<sample>/
```

| Setting in the paper | `data.dataset` | source directory scanned |
|---|---|---|
| CASME II, 3-class (MEGC2019 mapping) | `casme2_3c` | `casme2_7c` |
| CASME II, 5-class | `casme2_5c` | `casme2_7c` |
| SAMM, 5-class | `samm_5c` | `samm_8c` |
| SMIC-HS, 3-class (native) | `smic_hs_3c` | `smic_hs_3c` |
| MEGC2019 composite (CD) | `megc_cd_3c` | `casme2_7c` + `samm_8c` + `smic_hs_3c` |

* Label sets and merge rules are implemented in `src/data/dataset.py` (`DATASET_MERGE_CONFIG`) and
  documented in `src/data/DATASET.md`; the label set of each setting is also stated in the caption of
  Table 1 of the paper.
* Benchmark access: CASME II, SAMM and SMIC-HS are distributed by their original providers; we do not
  redistribute them. The CD setting is built from the 3-class subsets of the three.
* Class counts, held-out subjects and the exact number of samples per setting are recorded in
  [`release/splits/`](#released-artefacts) and are re-derived from the data directories at runtime.

## 2. Protocol (identical for every number in the paper)

| Aspect | Setting |
|---|---|
| Split | **LOSO** (leave-one-subject-out); one fold per subject |
| Validation | in-fold validation set, drawn from the *training* subjects, used **only** for epoch selection / early stopping (the test subject never participates in model selection) |
| Seeds | `1024 / 2048 / 4096`; every main number is reported as **mean ± SD** over seeds |
| Metrics | **pooled**: per-fold confusion matrices are summed *before* computing Accuracy, UF1 (macro-F1) and UAR (macro-recall); UF1 is the primary metric |
| Input | `rgb_dual_flow` = bidirectional dense optical flow (onset→apex and apex→offset) + apex RGB |

The protocol is configured in `src/config/`; `run_main.py` holds the run matrix (datasets × seeds ×
input representations). Configuration priority is `*`-suffixed key > command line > `init_config` >
model YAML > `common.yml`.

### Reproducing the comparison baselines

The literature values in the comparison tables are quoted, not measured here. The scripts can
likewise instantiate the compared methods (`HTNet`, `MPFNet`, `VITSRMCL`, `AlexNet`, `GoogLeNet`,
`VGG16`, `MMNet`, …) with `BPTNET_MODELS=HTNet`, but **a quoted value is not a like-for-like
comparison**: only a re-run inside this pipeline is, and the paper says so explicitly.

## 3. Outputs

```
saved/model/<model>/<dataset>/loso_<frame_type>_seed<seed>/
  results/cv_results.json      per-fold metrics + mean ± SD
  results/val_curve.json       validation curves
  checkpoints/best_model.pth   val-best checkpoint
```

## 4. Regenerating the tables

```bash
python tools/export_phase2.py --log log/metrics.jsonl --out phase2_numbers.json   # raw -> self-contained JSON
python tools/_fill_numbers.py --in phase2_numbers.json --apply                    # JSON -> paper number macros
python tools/_main_table.py      # main comparison tables (BPTNET_MAIN_FRAME=dual for the two-phase setting)
python tools/_ablation_table.py  # ablation table
python tools/pooled_metrics.py   # pooled Acc / UF1 / UAR from per-fold confusion matrices
python tools/_significance_ablation.py   # Welch t-tests reported with the ablation
```

`tools/` is a curated subset of the scripts that produced the manuscript; development-only diagnostics
are not included. The scripts are deliberately dependency-light: standard library + `numpy`. Every
number in the paper can be regenerated end to end with them, starting from the raw per-fold results.

## 5. Released artefacts

```
release/
  splits/loso_<setting>.json        fold -> test subject, training subjects, per-fold sample counts
  sample_lists/<setting>.csv        per-sample list (subject, sample, label, test fold)
  confusion_matrices/<run>.json     per-fold confusion matrices + pooled metrics recomputed from them
  README.md                         definition of every field and the self-check procedure
```

They are generated by `tools/make_release_artifacts.py`, which prefers the pipeline's own split
caches so that the released lists are exactly the samples each run used:

```bash
python tools/make_release_artifacts.py --selftest                     # verify folds/N against the paper
python tools/make_release_artifacts.py --data-dir <data-root> --log log/metrics.jsonl
```

Every pooled statistic in the paper can be recomputed from `release/confusion_matrices/`.

## 6. Repository layout

```
main.py                 training / evaluation entry point
run_main.py             experiment matrix (datasets × seeds × input representations)
src/                    data loading, models, evaluation, configuration
tools/                  table-regeneration and release scripts (see §4, §5)
release/                released splits, sample lists, per-fold confusion matrices
requirements.txt        pinned environment
```

## 7. Licence and citation

Code is released for research use. Benchmark images and optical flow are **not** redistributed;
please obtain them from the original providers and cite the original papers.

If you use this code, please cite the manuscript above. Contact: the corresponding author listed in
the paper.
