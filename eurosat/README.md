# Land Use Watch: explainable land-use classification and change detection on EuroSAT

Sentinel-2 satellite patches → 10 land-use classes → **why** the model confuses
certain classes → **where** land use changed between two dates.

## Research framing

Bare classification on EuroSAT is a solved benchmark. The original paper
already reports about 98.6% with a fine-tuned ResNet-50 on RGB. So "we trained a CNN on
EuroSAT" is not a contribution. This project uses classification as the base
layer and asks three questions on top of it:

| # | Question | Script | Evidence produced |
|---|----------|--------|-------------------|
| RQ1 | Which class pairs remain confused, and **why**: the input data or the model? | `confusability.py` | Grad-CAM grids, attention-focus statistic, colour/texture overlap (Bhattacharyya), embedding separation + PCA |
| RQ2 | Does the infrared information that the RGB release throws away resolve those confusions? | `train.py --composite ms_*` + `compare.py` | Same split, backbone and recipe; only the 3 input bands change (true colour vs NIR false colour vs SWIR) |
| RQ3 | How reliable is tile-level post-classification change detection, and how should the confidence threshold be set? | `change_detection.py --synthetic` | Precision / recall / false-alarm rate vs threshold on scenes with **known** ground-truth change |

Then it is applied to real before/after Sentinel-2 imagery, for example urban growth around
a city or forest loss, and served in a web app.

### Anchor literature

- P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep
  Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS,
  12(7), 2019. **The dataset paper. Your baseline numbers come from here.**
- R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization.* ICCV 2017. **The explanation method.**
- A. Singh. *Digital change detection techniques using remotely-sensed data.*
  Int. J. Remote Sensing, 10(6), 1989. **Classic review that defines
  post-classification comparison, the change-detection method used here.**
- M. Sandler et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
  CVPR 2018. **The default backbone.**

Check the exact citation details yourself before you submit. For related work, search Google Scholar for
papers that cite the EuroSAT paper and mention "explainability", "Grad-CAM" or
"change detection". Several 2023–2026 papers do this, and you need to position your work against them.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Class order, colours, paths, band definitions. Shared by everything. |
| `data.py` | Downloads EuroSAT (RGB or 13-band MS), makes the **fixed stratified 70/15/15 split** (`outputs/split.csv`), caches images as arrays. |
| `model.py` | Backbone + head (preprocessing is built into the model), fine-tuning helper, Grad-CAM. |
| `train.py` | Two-phase transfer learning: frozen head, then fine-tuning. Rotation/flip augmentation only. |
| `evaluate.py` | Test accuracy, macro-F1, per-class report, confusion matrix, ranked confused pairs, optional 8-view TTA. |
| `confusability.py` | RQ1 analysis per class pair. |
| `compare.py` | Results table across runs (RQ2). |
| `change_detection.py` | Tiling, classification on both dates, change map, hectares per transition, synthetic benchmark (RQ3). |
| `app.py` + `index.html` | Web app: classify + Grad-CAM, and before/after change detection. |
| `EuroSAT_Colab.ipynb` | Runs the whole pipeline on a free Colab GPU. |

## How to run

### Train in Colab (recommended)

Open `EuroSAT_Colab.ipynb` in Colab and pick a T4 GPU. It clones this branch, downloads
the data, trains, evaluates, runs the confusability study and the change benchmark,
and saves everything to Google Drive.

### Locally

```bash
cd eurosat
pip install -r requirements.txt
python data.py                          # download + split (≈90 MB RGB)
python train.py                         # RGB, MobileNetV2, 224 px
python evaluate.py --run outputs/rgb_mobilenetv2 --tta
python confusability.py --run outputs/rgb_mobilenetv2 --top 3
python confusability.py --run outputs/rgb_mobilenetv2 --pair AnnualCrop PermanentCrop
python change_detection.py --run outputs/rgb_mobilenetv2 --synthetic --scenes 100
python app.py                           # http://localhost:5000
```

Spectral experiment (RQ2, about 2 GB download):

```bash
for c in ms_rgb ms_falsecolor ms_swir; do
  python train.py --composite $c && python evaluate.py --run outputs/${c}_mobilenetv2
done
python compare.py outputs/ms_rgb_mobilenetv2 outputs/ms_falsecolor_mobilenetv2 outputs/ms_swir_mobilenetv2
```

Compare `ms_rgb` against the other two, not `rgb`. The official RGB JPEGs went
through different processing, so only `ms_rgb` isolates the effect of the bands.

### Real before/after imagery

1. Go to [Copernicus Browser](https://browser.dataspace.copernicus.eu), choose Sentinel-2 L2A
   and the True Color visualisation, and pick dates with less than 10% cloud.
2. Pick **the same month** in two different years. Otherwise seasonal crop cycles
   show up as false "changes".
3. Export both images with the **same bounding box** at **10 m resolution**.
4. Run `python change_detection.py --run outputs/rgb_mobilenetv2 --before a.png --after b.png --gsd 10`,
   or use the "Detect change" tab in the app.

## Design decisions worth defending in the viva

- **One fixed split for every experiment.** Differences between runs come from
  the model and the input, not from a different test set.
- **Dihedral augmentation only (flips + 90° rotations), no colour jitter.**
  Satellite images have no "up". But colour *is* the signal for crop types, so
  teaching the model to ignore it would undermine RQ1 and RQ2.
- **64 px upsampled to 224 px.** At native 64 px, MobileNetV2's final feature map
  is 2×2, which makes Grad-CAM useless. At 224 px it is 7×7.
- **Small fine-tuning learning rate (1e-5) with BatchNorm frozen.** In testing, a 10× larger LR
  made accuracy collapse as soon as the backbone was unfrozen.
- **Confidence-gated change detection.** With per-tile accuracy p, a naive "labels
  differ" rule flags about 1 − p² of *unchanged* tiles. At 97% accuracy, that is 6% false
  alarms. The synthetic benchmark measures how much the threshold reduces this.
- **Synthetic change benchmark built only from test patches.** Real change ground truth is scarce, and
  stitching held-out patches gives exact labels without leaking training data.
  Unchanged cells use a *different* patch of the same class on the second date,
  so the detector is not rewarded for matching identical pixels.

## Honest limitations (put these in the report)

- **Coarse resolution.** One tile is 640 m × 640 m (41 ha). Change smaller than that, such as a
  single building, is invisible, and mixed tiles get one label.
- **Domain shift.** EuroSAT contains only European scenes. Indian cropping patterns,
  building styles and vegetation look different, so expect lower accuracy on imagery
  from India. *Measuring* that drop on a few hand-labelled tiles is itself a valid
  finding.
- **Synthetic scenes are cleaner than reality.** They contain no registration errors,
  haze or seasonal drift, so treat the benchmark as an upper bound.
- **Grad-CAM is coarse (7×7) and only approximate.** Back qualitative claims with the
  quantitative statistics in `confusability.py`, not with heatmaps alone.
