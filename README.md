# Tomato bacterial leaf spot — VNIR days 0–3

Early (pre-symptomatic) detection of tomato bacterial leaf spot from VNIR hyperspectral leaf scans.

Dataset: Zhang, Vinatzer & Li 2024, *Hyperspectral imaging analysis for early detection of tomato bacterial leaf spot disease*, Scientific Reports 14:27666.

This build covers **days 0–3 only** (time points `bi`, `2hr`, `d1`, `d2`, `d3`). Visible spots start on day 4; no leaf in this range has visible spots. Pixel maps are **exploratory spectral difference from a healthy reference**, never “disease area”.

## Setup

Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the five scan pairs (`.bil` + `.bil.hdr`) in `data/raw/`:

| Stem | Time point | Run | Scan |
| --- | --- | --- | --- |
| `0dai_bi_visnir_bc_run2_1` | bi | 2 | 1 |
| `0dai_2hr_visnir_bc_run2-1` | 2hr | 2 | 1 |
| `1dai_visnir_bc_run1` | d1 | 1 | — |
| `2dai_visnir_bc_run1` | d2 | 1 | — |
| `3dai_visnir_bc_run1` | d3 | 1 | — |

Any other file in that folder (other camera, day 4+) is listed as skipped in the inventory and is never loaded or trained on.

SAM2 is optional (`segmentation.use_sam2: false` in `config.yaml`). The pipeline runs without it.

## Script order

Every script reads `config.yaml`. Re-run with `--force` to overwrite existing outputs.

1. `python scripts/run_inventory.py` — pair files, read headers only, write `outputs/inventory.csv`.
2. `python scripts/run_segment.py` — NDVI+Otsu leaves, overlays, per-leaf `.npz` caches, `outputs/leaves.csv`.
3. `python scripts/make_leaf_map_template.py` — write `data/leaf_map.csv` (blank `unit`/`group`). Optional `--assume-order` fills `unit = leaf_no` **only** when the leaf count equals that experiment’s unit count; every such row is tagged `assumed order, unconfirmed`.
4. **Fill `data/leaf_map.csv` by hand** (or confirm assumed order). Training will stop if any included leaf has no group, or if a training scan contains only one group.
5. `python scripts/run_features.py` — vegetation indices + pixel stats → `outputs/features.csv`.
6. `python scripts/run_explore.py` — spectra, index charts, PCA, Mann–Whitney tests.
7. `python scripts/run_train.py` — LightGBM + logistic regression + LDA, leave-one-unit-out, bi check, permutation test.
8. `python scripts/run_spread.py` — early-change distances and charts.
9. `streamlit run app/streamlit_app.py` — UI over a local folder or a zip of two scans.

Tests: `pytest tests/test_filenames.py tests/test_features.py`

## What the outputs mean

| Path | Meaning |
| --- | --- |
| `outputs/inventory.csv` | Every scan found, in-scope or skipped, plus header metadata |
| `outputs/overlays/<stem>.png` | False-colour image with numbered leaf outlines |
| `outputs/cache/<stem>__leafNN.npz` | Eroded-mask pixel spectra and coordinates |
| `outputs/leaves.csv` | One row per leaf (geometry + parsed filename fields) |
| `data/leaf_map.csv` | **Manual** experiment / unit / group / include |
| `outputs/features.csv` | Leaf metadata + indices + downsampled mean spectrum |
| `outputs/reports/index_tests.csv` | Per time point, inoculated vs control, BH-corrected p and Cliff’s delta |
| `outputs/reports/leaf_model.md` | Classifier report (expect modest early-stage accuracy) |
| `outputs/reports/model_comparison.csv` | LightGBM vs logistic vs LDA, including per-time-point and bi check |
| `outputs/reports/oof_predictions.csv` | Out-of-fold P(infected) |
| `outputs/models/leaf_model.joblib` | Selected model (highest OOF balanced accuracy; ties → simpler) |
| `outputs/reports/spread.md` | Early-change summary; not a disease-area map |
| `outputs/figures/heatmaps/` | Per-leaf Mahalanobis heatmaps vs healthy reference |

All paths, wavelengths, thresholds, and the run→experiment map live in `config.yaml`. There are no magic numbers in code.

## Design notes that follow from this subset

- **One scan per time point.** Every per-time-point metric is printed with its `n`.
- **Day 0 scans are run 2; days 1–3 are run 1.** If run = experiment, time point and run are confounded. Do not read a run difference as an infection effect, and do not train on one run and test on the other.
- **`bi` is never used for training.** It is the before-inoculation check and part of the healthy reference.
- **Small data** (expect roughly 40–50 leaves): small models, bootstrap CIs, permutation test.
- Crop margins in `config.yaml` are **starting values** for non-cloth strips; adjust after inspecting overlays.

## Out of scope (interfaces are kept general)

- Pixel-level bacterial-spot classifier and % spot area (needs visible spots, day 4+).
- Any data outside VNIR days 0–3. Time filters are config-driven.

## Open questions

1. What “run” means (experiment, batch, or repeat scan), and the `_1` / `-1` suffix. `run_to_experiment` in `config.yaml` defaults to `{1: 1, 2: 2}` and is **unconfirmed**.
2. Whether each scan mixes control and inoculated leaves, and the leaf order within each scan. Confirm from dataset metadata or the authors (Song Li, songli@vt.edu). Until then, `data/leaf_map.csv` must be filled by hand.
3. Whether the dataset includes per-leaf severity or bacterial count (CFU) tables.

## Citation

Zhang, X., Vinatzer, B.A. & Li, S. Hyperspectral imaging analysis for early detection of tomato bacterial leaf spot disease. *Sci Rep* **14**, 27666 (2024).

Li, S. (2024). Data from: Hyperspectral Imaging Analysis for Early Detection of Tomato Bacterial Leaf Spot Disease. Ag Data Commons. https://doi.org/10.15482/USDA.ADC/26046328
