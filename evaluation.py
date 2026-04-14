"""
evaluation.py — AUROC and pixel-level localization metrics

Two key metrics:
  - Image-level AUROC: how well anomaly scores separate normal vs. defective images
  - Pixel-level AUROC: how well heatmaps match ground truth defect masks
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def image_level_auroc(labels, scores):
    """
    Computes image-level AUROC.

    Args:
        labels (array): Ground truth — 0 for normal, 1 for anomalous
        scores (array): Predicted anomaly score per image (higher = more anomalous)

    Returns:
        float: AUROC score between 0 and 1 (1.0 = perfect)
    """
    return roc_auc_score(labels, scores)


def pixel_level_auroc(masks, heatmaps):
    """
    Computes pixel-level AUROC by flattening all masks and heatmaps.

    Args:
        masks (list of np.array): Ground truth binary masks (True = defect pixel)
        heatmaps (list of np.array): Predicted anomaly heatmaps (higher = more anomalous)

    Returns:
        float: Pixel-level AUROC score
    """
    all_masks = np.concatenate([m.flatten() for m in masks])
    all_heatmaps = np.concatenate([h.flatten() for h in heatmaps])
    return roc_auc_score(all_masks.astype(int), all_heatmaps)


def print_results(category, image_auroc, pixel_auroc=None):
    """Prints a clean results summary."""
    print(f'\n{"="*40}')
    print(f'Results — {category}')
    print(f'{"="*40}')
    print(f'Image-level AUROC: {image_auroc:.4f}')
    if pixel_auroc is not None:
        print(f'Pixel-level AUROC: {pixel_auroc:.4f}')
    print(f'{"="*40}\n')
