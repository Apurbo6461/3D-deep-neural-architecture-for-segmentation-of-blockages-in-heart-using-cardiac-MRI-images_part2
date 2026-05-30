import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: (B, C, D, H, W) logits
        target: (B, C, D, H, W) or (B, D, H, W)
        """
        # Apply sigmoid or softmax
        if pred.shape[1] > 1:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

def _binarize_for_metrics(tensor):
    """Ensure tensor is binary 0/1 for segmentation metrics (handles multi-class GT)."""
    t = tensor.view(-1).float()
    return (t > 0.5).float()


def _clamp_metric(value):
    """Clamp metric to [0, 1] and handle NaN."""
    if isinstance(value, torch.Tensor):
        value = value.item()
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def calculate_dice_score(pred, target, smooth=1e-5):
    """Calculate Dice score between prediction and target. Returns value in [0, 1]."""
    pred = pred.view(-1).float()
    # Binarize: pred already 0/1; target may be multi-class (0,1,2,...) -> treat >0.5 as foreground
    pred_bin = (pred > 0.5).float()
    target_bin = _binarize_for_metrics(target)

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()

    # Handle empty masks: if both pred and target are empty, return 1.0 (perfect agreement)
    # Otherwise, compute Dice normally
    if union.item() == 0:
        return 1.0 if intersection.item() == 0 else 0.0
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return _clamp_metric(dice)


def calculate_iou(pred, target, smooth=1e-5):
    """Calculate Intersection over Union (IoU). Returns value in [0, 1]."""
    pred_bin = _binarize_for_metrics(pred)
    target_bin = _binarize_for_metrics(target)

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if (union + smooth).item() <= 0:
        return 0.0
    iou = (intersection + smooth) / (union + smooth)
    return _clamp_metric(iou)


def calculate_accuracy(pred, target):
    """Calculate pixel-wise accuracy (binary: match after binarizing target). Returns value in [0, 1]."""
    pred_bin = (pred.view(-1).float() > 0.5).float()
    target_bin = _binarize_for_metrics(target)

    correct = (pred_bin == target_bin).sum()
    total = pred_bin.numel()
    if total == 0:
        return 0.0
    return _clamp_metric(correct.float() / total)


def calculate_sensitivity(pred, target, smooth=1e-5):
    """Calculate sensitivity (recall) for foreground class. Returns value in [0, 1]."""
    pred_bin = _binarize_for_metrics(pred)
    target_bin = _binarize_for_metrics(target)

    true_positives = ((pred_bin == 1) & (target_bin == 1)).sum()
    false_negatives = ((pred_bin == 0) & (target_bin == 1)).sum()

    denom = true_positives + false_negatives + smooth
    if denom.item() <= 0:
        return 0.0
    sensitivity = (true_positives + smooth) / denom
    return _clamp_metric(sensitivity)


def calculate_specificity(pred, target, smooth=1e-5):
    """Calculate specificity for background class. Returns value in [0, 1]."""
    pred_bin = _binarize_for_metrics(pred)
    target_bin = _binarize_for_metrics(target)

    true_negatives = ((pred_bin == 0) & (target_bin == 0)).sum()
    false_positives = ((pred_bin == 1) & (target_bin == 0)).sum()

    denom = true_negatives + false_positives + smooth
    if denom.item() <= 0:
        return 0.0
    specificity = (true_negatives + smooth) / denom
    return _clamp_metric(specificity)
