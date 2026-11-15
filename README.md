# Code for "Deep Learning–Based Analysis of Wound Closure Dynamics Identifies Cellular Senescence in Human Dermal Fibroblasts"

Analysis code for the P5/P15 scratch assay images. Only the scripts behind analyses reported in the paper are included.
Comments in the scripts are section titles only; printed output is in Korean. Images and CSV files are not in this repository.

## Scripts

| Paper | Content | Script (`src/`) |
|---|---|---|
| Methods: image preprocessing, deep learning | grayscale, flat-field, CLAHE; 0/24/48 h as 3 channels; ResNet18, plate-grouped CV | `build_cache.py`, `train_p5p15.py` |
| Sec. 2.5, Fig. 4f,g (raw-image and 2D-mask CNN) | within- and cross-experiment evaluation | `build_cache_new.py`, `train_crossds.py` |
| Sec. 2.5, Fig. 4f,g (row-wise mask CNN) | row-wise mask cache and evaluation | `build_cache_mask2.py`, `train_mask2.py` |
| Sec. 2.3, Fig. 3 | row-wise segmentation, wound width, closure fraction | `wound_width.py` (writes `wound_width.csv`) |
| Sec. 2.4, Fig. 4a-d | ridge-filter cell indicator, thresholds 0.30R / 0.55R | `seg_methods.py` |
| Methods: mask-based and scalar scores | 2D threshold mask (threshold from the 0 h image) | `seg_new.py` |
| Sec. 2.5, Fig. 4f,g (scalar model) | logistic regression on 48 h closure | `model_tabular.py` |
| Sec. 2.5, Fig. S1a | brightness, contrast, blue cast, Laplacian-variance sharpness | `audit_acquisition.py` |
| Sec. 2.5, Fig. S1b,c | blur matching and re-measured closure | `exp_blur_bias.py` |
| Sec. 2.3-2.4, Table 2 | width and closure statistics (standard library only) | `table2_stats.py` |
| Sec. 2.6, Fig. 5 (P7/P12 scoring) | [TO BE ADDED] | |

In the scripts and CSV outputs, Experiment 1 is dataset `OLD` (170 sets) and Experiment 2 is `NEW` (89 sets).

## Usage

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/images      # Windows: set DATA_ROOT=D:\path\to\images
```

Paths are set in `src/config.py`. Under `DATA_ROOT`, Experiment 1 is in `p5/` and `p15/26.09.29 hdf p 15 migration/`,
Experiment 2 in `20261013,4 AI hDF P5,15 migration(additional_data)/`, each laid out as `{0h,24h,48h}/<plate>/<position>.tif`.
A set is one plate and position imaged at all three time points (matched by the number at the end of the file name).
Everything the scripts write goes to `<DATA_ROOT>/work/`.

Run order: `wound_width.py`; `build_cache_new.py` and `build_cache_mask2.py`, then `train_crossds.py` and `train_mask2.py`; `model_tabular.py`; `exp_blur_bias.py`.
`table2_stats.py` reads `work/wound_width.csv`, or the path given as its argument.

## Data

Raw TIFFs (777 images, 259 sets): [TO BE FILLED]

## License / citation

[TO BE FILLED]
