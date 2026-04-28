"""
Optimized dataset loader for deepfake detection training.

Key improvements:
  - Stronger augmentation pipeline with RandomErasing and RandomPerspective
  - Adaptive subset sizing: uses 100% of data if dataset is small (<5k), 
    otherwise caps at reasonable limits for CPU training
  - Pin memory for faster data transfer to GPU
  - Reproducible splits with fixed random seed
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import numpy as np
from PIL import Image, ImageFilter
import io
import random


class RandomJPEGCompression:
    """Simulate JPEG compression artifacts at random quality levels."""
    def __init__(self, quality_lower=40, quality_upper=100):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper

    def __call__(self, img):
        if random.random() < 0.5:
            quality = random.randint(self.quality_lower, self.quality_upper)
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality)
            output.seek(0)
            return Image.open(output).convert('RGB')
        return img


class RandomDownscaleUpscale:
    """Simulate low-res image artifacts by downscaling then upscaling."""
    def __init__(self, min_scale=0.5, max_scale=0.8):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        if random.random() < 0.3:
            w, h = img.size
            scale = random.uniform(self.min_scale, self.max_scale)
            small = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return img


def get_dataloaders(data_dir, batch_size=32, img_size=224, use_sampler=True):
    """
    Expects dataset structure:
    data_dir/
      train/
        real/
        fake/
      val/
        real/
        fake/
    
    Returns train_loader, val_loader, classes
    """
    
    # Strong augmentation pipeline for robust training
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # slightly larger for random crop
        transforms.RandomCrop((img_size, img_size)),
        RandomJPEGCompression(40, 100),
        RandomDownscaleUpscale(0.5, 0.8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),  # cutout-style regularization
    ])

    # Clean preprocessing for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directories not found. Please ensure {train_dir} and {val_dir} exist.")

    train_dataset_full = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"[Dataset] Full train set: {len(train_dataset_full)} images")
    print(f"[Dataset] Full val set:   {len(val_dataset_full)} images")
    print(f"[Dataset] Classes: {train_dataset_full.classes}")

    # Reproducible splits
    rng = np.random.RandomState(42)

    # ── Adaptive subset sizing ────────────────────────────────
    # Small datasets (<5k): use 100%
    # Medium datasets (5k-50k): use 50%
    # Large datasets (>50k): use 20%
    total_train = len(train_dataset_full)
    if total_train < 5000:
        train_ratio = 1.0
    elif total_train < 50000:
        train_ratio = 0.50
    else:
        train_ratio = 0.20

    total_val = len(val_dataset_full)
    val_ratio = min(1.0, max(0.20, 5000 / max(total_val, 1)))  # at least 20%, aim for ~5k samples

    # Stratified sampling: equal per class
    targets = np.array(train_dataset_full.targets)
    classes = np.unique(targets)
    
    train_indices = []
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        n_select = max(1, int(len(cls_idx) * train_ratio))
        n_select = min(n_select, len(cls_idx))
        selected = rng.choice(cls_idx, n_select, replace=False)
        train_indices.extend(selected)
    
    rng.shuffle(train_indices)
    train_indices = np.array(train_indices)

    # Validation subset
    val_targets = np.array(val_dataset_full.targets)
    val_indices = []
    for cls in np.unique(val_targets):
        cls_idx = np.where(val_targets == cls)[0]
        n_select = max(1, int(len(cls_idx) * val_ratio))
        n_select = min(n_select, len(cls_idx))
        selected = rng.choice(cls_idx, n_select, replace=False)
        val_indices.extend(selected)
    val_indices = np.array(val_indices)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"[Dataset] Using {len(train_indices)} train / {len(val_indices)} val samples "
          f"({train_ratio*100:.0f}% / {val_ratio*100:.0f}%)")

    # Count per class
    train_class_counts = {}
    for idx in train_indices:
        cls = train_dataset_full.targets[idx]
        cls_name = train_dataset_full.classes[cls]
        train_class_counts[cls_name] = train_class_counts.get(cls_name, 0) + 1
    for name, count in sorted(train_class_counts.items()):
        print(f"[Dataset] Train - {name}: {count}")

    # Weighted sampler for class balance
    if use_sampler:
        sample_targets = np.array([train_dataset_full.targets[i] for i in train_indices])
        class_counts = np.array([len(np.where(sample_targets == t)[0]) for t in np.unique(sample_targets)])
        weight = 1. / class_counts
        samples_weight = np.array([weight[t] for t in sample_targets])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=2, pin_memory=(torch.cuda.is_available())
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=(torch.cuda.is_available())
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(torch.cuda.is_available())
    )

    return train_loader, val_loader, train_dataset_full.classes
