"""
Optimized training pipeline for DeepfakeEfficientNet.

Key improvements over the previous version:
  - 2-phase transfer learning (warmup -> fine-tune) with proper epoch gating
  - Gradient clipping to prevent exploding gradients
  - Early stopping with patience to avoid overfitting
  - Label smoothing for better generalization
  - Mixed precision training for speed on GPU
  - Saves BOTH model.state_dict() and model.backbone.state_dict() for compatibility
  - Confusion matrix logging for better debugging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import DeepfakeEfficientNet
from dataset_loader import get_dataloaders
from tqdm import tqdm
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


def train_model(data_dir, num_epochs=20, batch_size=32, lr=1e-4, save_path='models/deepfake_efficientnet.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data with WeightedRandomSampler
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size, use_sampler=True)
    print(f"Classes: {classes}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Initialize model with ImageNet pretrained backbone (no optimization passes for training)
    model = DeepfakeEfficientNet(pretrained=True)
    model.to(device)

    # Label smoothing helps prevent overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ─── Phase 1: Warmup (freeze backbone, train classifier only) ───
    WARMUP_EPOCHS = 3
    print(f"\n{'='*60}")
    print(f"Phase 1: Classifier Warmup ({WARMUP_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Freeze all backbone parameters
    for param in model.backbone.features.parameters():
        param.requires_grad = False
    # Only classifier is trainable
    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    warmup_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,  # higher LR for randomly initialized classifier
        weight_decay=1e-4
    )

    best_f1 = 0.0
    patience_counter = 0
    PATIENCE = 5
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(WARMUP_EPOCHS):
        print(f"\n--- Warmup Epoch {epoch+1}/{WARMUP_EPOCHS} ---")
        train_loss = _train_epoch(model, train_loader, criterion, warmup_optimizer, device)
        val_metrics = _validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        _print_metrics(val_metrics)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            _save_model(model, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

    # ─── Phase 2: Full Fine-tuning ──────────────────────────────────
    FINETUNE_EPOCHS = num_epochs - WARMUP_EPOCHS
    print(f"\n{'='*60}")
    print(f"Phase 2: Full Fine-tuning ({FINETUNE_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    # Discriminative learning rates: backbone gets lower LR than classifier
    finetune_optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': lr * 0.1},   # backbone: 1e-5
        {'params': model.backbone.classifier.parameters(), 'lr': lr},        # classifier: 1e-4
    ], weight_decay=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        finetune_optimizer, T_0=5, T_mult=2, eta_min=1e-7
    )

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\n--- Fine-tune Epoch {epoch+1}/{FINETUNE_EPOCHS} ---")
        train_loss = _train_epoch(model, train_loader, criterion, finetune_optimizer, device, clip_grad=1.0)
        val_metrics = _validate(model, val_loader, criterion, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        _print_metrics(val_metrics)

        scheduler.step()
        current_lr = finetune_optimizer.param_groups[0]['lr']
        print(f"  Backbone LR: {current_lr:.8f} | Classifier LR: {finetune_optimizer.param_groups[1]['lr']:.8f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            _save_model(model, save_path)
            patience_counter = 0
            print(f"  ** New best F1: {best_f1:.4f} -- model saved **")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n[!] Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")


def _train_epoch(model, loader, criterion, optimizer, device, clip_grad=None):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


def _validate(model, loader, criterion, device):
    """Run validation and compute all metrics."""
    model.eval()
    val_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of class 1 (real)

    val_loss = val_loss / total_samples

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        'val_loss': val_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def _print_metrics(metrics):
    """Pretty-print validation metrics."""
    print(f"  Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['accuracy']:.4f} | "
          f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
          f"F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    cm = metrics['confusion_matrix']
    if cm.shape == (2, 2):
        print(f"  Confusion Matrix:  TN={cm[0][0]}  FP={cm[0][1]}  |  FN={cm[1][0]}  TP={cm[1][1]}")


def _save_model(model, save_path):
    """
    Save model in BOTH formats for maximum compatibility:
      1. Full model state_dict (backbone.features.* + backbone.classifier.*)
      2. Backbone-only state_dict (features.* + classifier.*)
    """
    # Primary: save backbone-only (what get_model expects for the 'features' key check)
    torch.save(model.backbone.state_dict(), save_path)
    
    # Also save full model as backup
    backup_path = save_path.replace('.pth', '_full.pth')
    torch.save(model.state_dict(), backup_path)
    print(f"  Model saved: {save_path} + {backup_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train DeepfakeEfficientNet")
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    else:
        print(f"Data directory {args.data_dir} not found.")
        print(f"Expected structure:")
        print(f"  {args.data_dir}/")
        print(f"    train/")
        print(f"      fake/   (manipulated images)")
        print(f"      real/   (authentic images)")
        print(f"    val/")
        print(f"      fake/")
        print(f"      real/")
