# Zero-Shot Industrial Anomaly Detection with CLIP + DINOv2

A research project exploring zero-shot anomaly detection in industrial images by combining CLIP's language-vision embeddings with DINOv2's spatially rich patch features to produce precise pixel-level anomaly heatmaps.

---

## Overview

Traditional anomaly detection requires thousands of labeled defect images. This project uses **CLIP** (OpenAI) to detect defects purely from text descriptions — no defective training examples needed. We then improve on the WinCLIP baseline by fusing **DINOv2** (Meta AI) patch features to produce accurate pixel-level localization heatmaps.

### Key ideas
- **Zero-shot**: the model never sees a defective image during setup
- **Text-driven**: you describe what "normal" and "anomalous" look like in plain English
- **Pixel-level**: outputs a heatmap highlighting exactly where the defect is

---

## Project Structure

```
clip-anomaly-detection/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── .gitkeep              # Download MVTec AD here (not included)
│
├── notebooks/
│   ├── 01_baseline_clip.ipynb       # Phase 1: Reproduce WinCLIP baseline
│   ├── 02_dinov2_fusion.ipynb       # Phase 2: Add DINOv2 spatial features
│   └── 03_experiments.ipynb         # Phase 3: Ablations + prompt study
│
├── src/
│   ├── __init__.py
│   ├── dataset.py            # MVTec AD dataloader
│   ├── clip_model.py         # CLIP loading + image/text scoring
│   ├── dinov2_model.py       # DINOv2 loading + patch feature extraction
│   ├── fusion.py             # Combining CLIP + DINOv2 scores
│   ├── evaluation.py         # AUROC + pixel localization metrics
│   └── visualize.py          # Heatmap generation and plotting
│
├── prompts/
│   └── bottle.json           # Text prompts per MVTec category
│
└── results/
    └── .gitkeep              # Heatmap outputs saved here locally
```

---

## Dataset

Download **MVTec AD** from the official source:
👉 https://www.mvtec.com/company/research/datasets/mvtec-ad

Place the extracted category folders inside the `data/` directory:

```
data/
├── bottle/
├── leather/
├── metal_nut/
└── ...
```

> MVTec AD is free for research use. Do not upload it to GitHub (4.9 GB).

---

## Setup

### Option A — Google Colab (recommended)
1. Open any notebook in `notebooks/` directly in Google Colab
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Mount your Google Drive where the MVTec data lives
4. Run the setup cell at the top of each notebook

### Option B — Local (NVIDIA GPU required)
```bash
git clone https://github.com/YOUR_USERNAME/clip-anomaly-detection.git
cd clip-anomaly-detection
pip install -r requirements.txt
```

---

## Baseline: WinCLIP

This project reproduces and improves on **WinCLIP**:
> Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023

Our contribution: fusing DINOv2 patch-level features with CLIP similarity scores to improve pixel-level localization on MVTec AD.

---

## Evaluation

| Metric | Description |
|---|---|
| AUROC (image-level) | How well the model separates normal vs. defective images |
| AUROC (pixel-level) | How accurately the heatmap matches ground truth defect masks |

---

## Models Used

| Model | Source | Role |
|---|---|---|
| CLIP ViT-B/16 | OpenAI via open-clip-torch | Image-text similarity scoring |
| DINOv2 ViT-S/14 | Meta AI via torch.hub | Spatial patch feature extraction |

---

## Results

*To be updated as experiments are completed.*

| Method | Image AUROC | Pixel AUROC |
|---|---|---|
| WinCLIP (reproduced) | - | - |
| Ours (CLIP + DINOv2) | - | - |

---

## References

- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
- Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation", CVPR 2023
- Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019
