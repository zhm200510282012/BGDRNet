# BGDRNet

Source and numerical records for **Boundary-Guided Detail Routing for Parameter-Efficient Glass Surface Segmentation**.

Huangming Zhang, School of Electrical and Information Engineering, Jiangsu University, Zhenjiang 212013, China. Contact: huangming.zhang.scholar@outlook.com.

BGDRNet uses a truncated pretrained MobileNetV2 encoder and a 32-channel boundary-guided region-detail decoder. Its complete inference model has 278,804 parameters. `FR29` is the frozen implementation identifier for BGDRNet; `S0` denotes the basic decoder and `FR28` the shared-output-mapping control.

## Reproduce the archived main statistics

With Python 3.10 or later, from the repository root:

```sh
python recompute_statistics.py
```

This standard-library-only command checks all 12 main evaluation endpoints (two models, three seeds, two datasets). It verifies sample IDs and counts, recomputes pooled IoU, false-positive/negative rates and MAE from per-image records, averages archived boundary F1, and checks the paired mean and sample SD. Expected output: `"status": "PASS"`. It does not train a model, run inference, or reconstruct boundary F1 from masks.

## Files

| Path | Content |
|---|---|
| `jobs/S0_seed*`, `jobs/FR29_seed*` | Exact source closures and path-templated configurations for the six principal fits |
| `jobs/FR28_seed*` | Exact source closures/configurations for the three shared-mapping controls |
| `splits/` | Fixed membership: 2,168 training, 542 development, 813 GSD test and 950 GEGD2 images |
| `records/` | Principal training histories and selected-checkpoint metrics |
| `evaluation/` | Per-image sufficient statistics for the 12 principal evaluation endpoints |
| `evidence/`, `paper_tables/` | Archived summary and numerical tables |
| `secondary/` | External-comparator and local-control sources and archived records, including GlassNet |
| `checkpoint_index.json` | Model identifiers, selected epochs and checkpoint inventory; weights are not included |

All files are individually available in this repository; the implementation is not supplied only as a compressed archive.

## Data and exact initial weights

Obtain GSD from [the author project](https://jiaying.link/cvpr2021-gsd/) and GEGD2 from [GhostingNet](https://github.com/YT3DVision/GhostingNet). The GEGD2 archive is the release identifier used here for the glass annotations. Dataset images and masks are not redistributed.

Arrange GSD as `train/image/<id>.jpg`, `train/mask/<id>.png`, `test/image/<id>.jpg`, and `test/mask/<id>.png` beneath a chosen data root. Follow the filenames and extensions in `splits/`; do not regenerate the split. Keep GEGD2 as its original ZIP, with member names matching `splits/gegd2.csv`. GSD foreground is mask > 0; GEGD2 foreground is mask == 255. Preserve the official duplicated test entry.

Use the exact legacy [MobileNetV2 weights](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth), `mobilenet_v2-b0353104.pth`, rather than the current TorchVision DEFAULT weights. `pretrained_identity.json` records their identity. Third-party data and weights retain their original terms.

## Prepare and train a fresh main-model run

Recorded environment: Python 3.10.18, torch 2.7.1+cu126, torchvision 0.22.1+cu126, NumPy 2.1.2, Pillow 11.0.0, OpenCV 4.10.0; RTX 4070 Laptop GPU. Install the matching PyTorch build separately and consult `requirements-observed.txt`. A clean-environment training reproduction was not performed during release preparation.

Example paths below are placeholders for your data and output directories:

```sh
python prepare_run.py --job FR29_seed42 --gsd-root /path/to/GSD --pretrained /path/to/mobilenet_v2-b0353104.pth --out /path/to/new_run
cd /path/to/new_run
python worker.py --mode smoke
python worker.py --mode train
```

The preparation command refuses an existing output directory. Repeat with `S0_seed42`, seeds 43/44, or the `FR28_seed*` control jobs. It relocates configuration and smoke-reference paths while retaining model/loss/optimizer implementations. The correct entry point is `worker.py`; do not execute the historical multi-variant `experiment.py --mode run` entry point.

The protocol uses 40 epochs, batch size 8, 256 x 256 input, AdamW (lr 0.0003, weight decay 0.01), cosine decay, horizontal flips, BF16 autocast with FP32 weights, and frozen encoder BN running statistics. Select maximal development global mIoU, taking the earliest exact tie. Main-model selected epochs are 30/25/15 for S0 and 28/27/28 for FR29. S0 disables auxiliary heads/objectives through model options despite their weights appearing in the common configuration.

## Evaluate a deployment checkpoint

After training, use the resulting trusted deployment checkpoint:

```sh
python evaluate.py --job FR29_seed42 --dataset GSD_test --checkpoint /path/to/deployment.pt --gsd-root /path/to/GSD --out /path/to/new_eval_gsd
python evaluate.py --job FR29_seed42 --dataset GEGD2 --checkpoint /path/to/deployment.pt --gegd2-zip /path/to/GEGD2.zip --out /path/to/new_eval_gegd2
```

The wrapper uses strict state loading, threshold 0.5, native-resolution logit resizing and the original metric code. It refuses an existing output directory. Fitted weights are not included in this source release; fresh training is the documented route to a checkpoint. Model latency in the paper excludes image loading and native-resolution CPU postprocessing.

## Verification coverage

Release checks validate Python syntax, local import closure, fixed sample order, numerical aggregation and the archived main paired statistics. They do not constitute fresh model training or end-to-end inference. Historical source snapshots under `secondary/` retain their original path/configuration conventions and are accompanied by records; they are not a one-command portable runner for every table. In particular, external comparisons and timing tables are preserved records, not outputs recomputed by `recompute_statistics.py`.

The baseline comparison estimates the full trained system, including supervision and capacity changes. External methods differ in initialization/supervision; GlassNet is a separate high-capacity reference. The two evaluation datasets had been observed in earlier project stages. Three seeds and two datasets are not six independent training runs. See the manuscript for the full experimental boundaries.

## Reuse and citation

See `TERMS.txt` and the original third-party notices in `licenses/` and `secondary/`. No dataset or model-weight redistribution license is implied by a software dependency's license. References for datasets, backbones and comparators are provided in `references.bib`. The manuscript is being prepared for submission; no publication DOI is claimed.
