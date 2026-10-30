---
license: cc-by-nc-4.0
tags:
  - glass-surface-detection
  - segmentation
  - pytorch
datasets:
  - garrying/GSD
---

# GlassNet — pretrained on GSD

Pretrained checkpoint for **GlassNet** from the CVPR 2021 paper:

> **Rich Context Aggregation with Reflection Prior for Glass Surface Detection**  
> Jiaying Lin, Zebang He, Rynson W.H. Lau  
> *Proceedings of CVPR 2021*  
> Project page: https://jiaying.link/cvpr2021-gsd/

## Files

| File | Description |
|------|-------------|
| `GSD.pth` | GlassNet weights trained on the GSD training split |

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from model import GlassNet

ckpt = hf_hub_download("garrying/GSD-GlassNet", "GSD.pth")
net = GlassNet()
net.load_state_dict(torch.load(ckpt, map_location="cpu"))
net.eval()
```

For full inference code (data loading, CRF post-processing, saving outputs) see `infer.py` in the original release.

## Model Architecture

**GlassNet** uses a ResNeXt-101 backbone with:

- **DenseContrastModule** — multi-scale dilated convolutions (rates 1/2/4/8) with pairwise feature subtraction to capture cross-context contrast
- **SELayer** — grouped squeeze-and-excitation for context-aware channel reweighting
- **RefNet** — a lightweight U-Net-style decoder that jointly predicts the binary glass mask and reconstructs the reflection image as auxiliary output
- **CRF post-processing** — dense CRF refinement of predicted masks at inference time

## Dataset

The GSD dataset is available at [garrying/GSD](https://huggingface.co/datasets/garrying/GSD).

## Citation

```bibtex
@inproceedings{GSD:2021,
    title     = {Rich Context Aggregation with Reflection Prior for Glass Surface Detection},
    author    = {Lin, Jiaying and He, Zebang and Lau, Rynson W.H.},
    booktitle = {Proc. CVPR},
    year      = {2021}
}
```

## Contact

jiayinlin5-c@my.cityu.edu.hk
