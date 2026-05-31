"""Verification tests for the multi-class segmentation pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

print("=" * 60)
print("VERIFICATION TESTS — Multi-Class Segmentation Pipeline")
print("=" * 60)

# ------------------------------------------------------------------
# Test 1: Dataset returns class labels {0,1,2,3}
# ------------------------------------------------------------------
print("\n[1] Dataset test ...")
from data.dataset import MedicalDataset
ds = MedicalDataset("E:/Thesis Dataset 2/training")
img, gt = ds[0]
print(f"    img  shape={img.shape}  dtype={img.dtype}")
print(f"    gt   shape={gt.shape}   dtype={gt.dtype}")
print(f"    gt   unique={gt.unique().tolist()}")
assert img.ndim == 4 and img.shape[0] == 1, "img should be (1,H,W,D)"
assert gt.ndim  == 3,                       "gt should be (H,W,D) — no channel dim"
assert gt.dtype == torch.int64,             "gt dtype should be int64 (long)"
print("    PASS")

# ------------------------------------------------------------------
# Test 2: All 3 models produce (1, 4, H, W, D)
# ------------------------------------------------------------------
print("\n[2] Model output-shape test ...")
from models.unet3d_standard import UNet3D
from models.vnet             import VNet3D
from models.res_att_unet3d   import ResAttUNet3D

x = torch.randn(1, 1, 128, 128, 64)
expected = (1, 4, 128, 128, 64)
for Cls, name, kw in [
    (UNet3D,       "3D-U-Net",        {"base_filters": 32}),
    (VNet3D,       "V-Net",           {"base_filters": 16}),
    (ResAttUNet3D, "ResAtt-3D-U-Net", {"base_filters": 16}),
]:
    m = Cls(in_channels=1, out_channels=4, **kw)
    m.eval()
    with torch.no_grad():
        y = m(x)
    print(f"    {name:22s}  output={tuple(y.shape)}")
    assert tuple(y.shape) == expected, f"Expected {expected}, got {tuple(y.shape)}"
print("    PASS")

# ------------------------------------------------------------------
# Test 3: Loss functions
# ------------------------------------------------------------------
print("\n[3] Loss-function test ...")
from utils import CombinedLoss, MultiClassDiceLoss, calculate_per_class_dice

logits  = torch.randn(1, 4, 32, 32, 16)
targets = torch.randint(0, 4, (1, 32, 32, 16))

dice_loss = MultiClassDiceLoss()(logits, targets)
comb_loss = CombinedLoss()(logits, targets)
scores    = calculate_per_class_dice(logits, targets)

print(f"    MultiClassDiceLoss = {dice_loss.item():.4f}")
print(f"    CombinedLoss       = {comb_loss.item():.4f}")
print(f"    Per-class Dice     = {scores}")
assert 0 <= dice_loss.item() <= 2, "Dice loss out of expected range"
assert 0 <= comb_loss.item() <= 5, "Combined loss out of expected range"
print("    PASS")

# ------------------------------------------------------------------
# Test 4: pred_to_class_masks helper
# ------------------------------------------------------------------
print("\n[4] pred_to_class_masks test ...")
from utils import pred_to_class_masks
fake_logits = torch.randn(1, 4, 64, 64, 32)
masks = pred_to_class_masks(fake_logits)
print(f"    keys = {list(masks.keys())}")
for k, v in masks.items():
    print(f"    {k}: shape={v.shape}  unique={np.unique(v)}")
    assert set(np.unique(v)).issubset({0.0, 1.0}), "masks should be binary"
print("    PASS")

# ------------------------------------------------------------------
# Test 5: BlockageDetector — multiclass path
# ------------------------------------------------------------------
print("\n[5] BlockageDetector.detect_blockages_multiclass test ...")
from blockage_detection import BlockageDetector
bd = BlockageDetector()

ed_logits = torch.randn(4, 128, 128, 64)
result    = bd.detect_blockages_multiclass(ed_logits, ed_logits)

required_keys = [
    'blockage_rate', 'blockage_mask', 'severity', 'abnormality_score',
    'ef_pct', 'cavity_reduction_pct', 'mean_wall_thickness',
    'MYO', 'LV', 'RV', 'class_masks_ed',
]
for k in required_keys:
    assert k in result, f"Missing key: {k}"
print(f"    blockage_rate   = {result['blockage_rate']:.2f}%")
print(f"    abnormality     = {result['abnormality_score']}/5")
print(f"    LV has_blockage = {result['LV']['has_blockage']}")
print(f"    MYO blockage    = {result['MYO']['blockage_rate']:.2f}%")
print("    PASS")

# ------------------------------------------------------------------
# Test 6: AnatomicalRegionIdentifier — multiclass path
# ------------------------------------------------------------------
print("\n[6] AnatomicalRegionIdentifier.identify_regions_from_multiclass test ...")
from anatomical_region_identifier import AnatomicalRegionIdentifier
ari = AnatomicalRegionIdentifier()

regions = ari.identify_regions_from_multiclass(ed_logits)
print(f"    keys = {list(regions.keys())}")
for rname, mask in regions.items():
    print(f"    {rname:12s}: shape={mask.shape}  sum={int(mask.sum())}")
assert set(regions.keys()) == {'LV', 'RV', 'MYOCARDIAL', 'UNKNOWN'}
print("    PASS")

# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
