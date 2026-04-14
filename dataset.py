"""
dataset.py — MVTec AD dataloader

Loads images and ground truth masks from the MVTec AD folder structure:
    category/
    ├── train/good/
    ├── test/good/
    ├── test/<defect_type>/
    └── ground_truth/<defect_type>/
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np


class MVTecDataset:
    """
    Loads all test images and their ground truth labels for one MVTec category.

    Args:
        root_path (str): Path to the category folder, e.g. 'data/bottle'
        transform: Optional image transform (e.g. CLIP's preprocess function)
        mask_size (tuple): Size to resize ground truth masks to (H, W)
    """

    def __init__(self, root_path, transform=None, mask_size=(224, 224)):
        self.root_path = Path(root_path)
        self.transform = transform
        self.mask_size = mask_size
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        test_path = self.root_path / 'test'
        gt_path = self.root_path / 'ground_truth'

        for split_dir in sorted(test_path.iterdir()):
            is_anomaly = split_dir.name != 'good'

            for img_path in sorted(split_dir.glob('*.png')):
                # Find corresponding ground truth mask if anomalous
                mask_path = None
                if is_anomaly:
                    mask_path = gt_path / split_dir.name / (img_path.stem + '_mask.png')
                    if not mask_path.exists():
                        mask_path = None

                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'label': 1 if is_anomaly else 0,
                    'defect_type': split_dir.name,
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load mask (or blank mask for normal images)
        if sample['mask_path'] and sample['mask_path'].exists():
            mask = Image.open(sample['mask_path']).convert('L')
            mask = mask.resize(self.mask_size, Image.NEAREST)
            mask = np.array(mask) > 0  # binary: True where defect is
        else:
            mask = np.zeros(self.mask_size, dtype=bool)

        return {
            'image': image,
            'mask': mask,
            'label': sample['label'],
            'defect_type': sample['defect_type'],
            'image_path': str(sample['image_path']),
        }


def get_normal_train_images(root_path, transform=None):
    """
    Returns a list of normal training images.
    Used in Phase 2 for computing a 'normal' feature distribution.
    """
    train_path = Path(root_path) / 'train' / 'good'
    images = []
    for img_path in sorted(train_path.glob('*.png')):
        image = Image.open(img_path).convert('RGB')
        if transform:
            image = transform(image)
        images.append(image)
    return images
