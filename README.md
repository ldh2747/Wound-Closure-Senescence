# Code for "Deep Learning–Based Analysis of Wound Closure Dynamics Identifies Cellular Senescence in Human Dermal Fibroblasts"

Analysis code for the P5/P15 scratch-assay images. Only scripts that correspond to analyses reported in the paper are included.
Comments and console output inside the scripts are in Korean. Raw images are not included (see *Data*).

## Paper element -> script

| Paper | What | Script(s) in `src/` |
|---|---|---|
| Methods, *Image preprocessing*, *Deep learning* | grayscale + flat-field + CLAHE, 3 time points as channels; ResNet18 (AdamW, cosine, 25 epochs, batch 16); stratified plate-grouped CV; shared model/training code | `build_cache.py`, `train_p5p15.py` |
| Sec. 2.5, Fig. 4f,g (raw-image CNN, 2D-mask CNN) | within-experiment CV (4 folds Exp. 1, 2 folds Exp. 2) and cross-experiment testing | `build_cache_new.py` (raw and 2D-mask caches), `train_crossds.py` |
| Sec. 2.5, Fig. 4f,g (row-wise mask CNN) | row-wise mask cache and the same evaluation | `build_cache_mask2.py`, `train_mask2.py` |
| Sec. 2.3, Fig. 3; Methods, *Wound segmentation / Wound metrics* | row-wise segmentation (texture ratio), wound width and closure fraction per set | `wound_width.py` (writes `wound_width.csv`) |
| Sec. 2.4, Fig. 4a-d (ridge filter) | alternative cell indicator, thresholds 0.30R / 0.55R | `seg_methods.py` |
| Methods, *Mask-based and scalar scores* (2D threshold mask) | 2D threshold segmentation with a threshold set on the 0 h image | `seg_new.py` |
| Sec. 2.5, Fig. 4f,g (scalar model) | logistic regression on the 48 h closure fraction | `model_tabular.py` |
| Sec. 2.5, Fig. S1a; Methods, *Image quality control* | per-image brightness, contrast, blue cast, Laplacian-variance sharpness | `audit_acquisition.py` |
| Sec. 2.5, Fig. S1b,c; Methods, *simulated defocus* | sharpness-matched artificial blurring and re-measurement of closure | `exp_blur_bias.py` |
| Sec. 2.3-2.4, Table 2 | initial width, plate-level tests, width-adjusted difference, Spearman correlation | `table2_stats.py` (standard library only) |
| Sec. 2.6, Fig. 5 (P7/P12 scoring) | [TO BE ADDED] scoring script | – |

Experiment 1 = dataset `OLD` (170 sets), Experiment 2 = dataset `NEW` (89 sets) in the scripts and in the CSV outputs.

## Setup

```bash
pip install -r requirements.txt
export MIDAS_DATA_ROOT=/path/to/images        # Windows: set MIDAS_DATA_ROOT=D:\path\to\images
```

All paths are defined in `src/config.py`. Expected layout under `MIDAS_DATA_ROOT`: Experiment 1 in `p5/` and `p15/26.09.29 hdf p 15 migration/`,
Experiment 2 in `20261013,4 AI hDF P5,15 migration(additional_data)/`, each as `{0h,24h,48h}/<plate>/<position>.tif`.
Outputs go to `<MIDAS_DATA_ROOT>/work/`. A set = same plate and position imaged at all three time points (positions matched by the trailing number of the file name).
Typical order: `wound_width.py`; `build_cache_new.py`, `build_cache_mask2.py` -> `train_crossds.py`, `train_mask2.py`; `model_tabular.py`; `exp_blur_bias.py`.

## Data

This repository contains code only; no images and no CSV files. All CSV files are written by the scripts to `<MIDAS_DATA_ROOT>/work/`.
`table2_stats.py` reads `work/wound_width.csv` (or a path given as argument).

Raw TIFFs (777 images in 259 sets): [TO BE FILLED: availability].

## License / citation

[TO BE FILLED by the authors]
