"""
Blockage Detection Demo: 3 models × 5 samples each.
Runs segmentation + blockage detection for 3D U-Net, V-Net, and ResAtt-3D-U-Net
on 5 test samples per model, saves visualizations and a summary.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
os.chdir(_script_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from data.dataset import MedicalDataset
from models.unet3d_standard import UNet3D
from models.vnet import VNet3D
from models.unet3d import ResAttUNet3D
from blockage_detection import BlockageDetector

# Config
TEST_DIR = "/Users/abidhasan/Downloads/for thesis/dataset copy 6/testing"
TARGET_SHAPE = (128, 128, 64)
SAMPLES_PER_MODEL = 5

MODEL_CONFIGS = {
    '3D U-Net': {
        'class': UNet3D,
        'params': {'in_channels': 1, 'out_channels': 1, 'base_filters': 32},
        'path': os.path.join(_script_dir, 'best_3d_u-net.pth')
    },
    'V-Net': {
        'class': VNet3D,
        'params': {'in_channels': 1, 'out_channels': 1, 'base_filters': 16},
        'path': os.path.join(_script_dir, 'best_v-net.pth')
    },
    'ResAtt-3D-U-Net': {
        'class': ResAttUNet3D,
        'params': {'in_channels': 1, 'out_channels': 1, 'base_filters': 16},
        'path': os.path.join(_script_dir, 'best_resatt-3d-u-net.pth')
    }
}


def load_models(device):
    """Load all 3 models."""
    models = {}
    for name, config in MODEL_CONFIGS.items():
        model = config['class'](**config['params'])
        if os.path.exists(config['path']):
            model.load_state_dict(torch.load(config['path'], map_location=device))
            print(f"  ✓ Loaded {name}")
        else:
            print(f"  ⚠ Weights not found for {name}, using random init")
        model = model.to(device)
        model.eval()
        models[name] = model
    return models


def run_blockage_detection_demo():
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading models...")
    models = load_models(device)

    print(f"Loading test dataset from {TEST_DIR}...")
    test_dataset = MedicalDataset(root_dir=TEST_DIR, target_shape=TARGET_SHAPE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    detector = BlockageDetector(threshold_thickening=0.3, min_blockage_size=5)
    all_results = []

    for model_name, model in models.items():
        print(f"\n--- {model_name} (5 samples) ---")
        sample_count = 0

        with torch.no_grad():
            for batch_idx, (imgs, gts) in enumerate(tqdm(test_loader, desc=model_name, total=SAMPLES_PER_MODEL)):
                if sample_count >= SAMPLES_PER_MODEL:
                    break

                imgs = imgs.to(device)
                img_np = imgs[0, 0].cpu().numpy()
                gt_np = gts[0, 0].numpy()

                preds = model(imgs)
                if preds.shape[1] == 1:
                    preds_binary = (torch.sigmoid(preds) > 0.5).float()
                else:
                    preds_binary = (torch.softmax(preds, dim=1)[:, 0:1] < 0.5).float()
                pred_np = preds_binary[0, 0].cpu().numpy()

                blockage_info = detector.detect_blockages(pred_np, img_np)
                blockage_mask = blockage_info['blockage_mask']

                result = {
                    'model_name': model_name,
                    'sample_id': batch_idx,
                    'sample_index': sample_count + 1,
                    'blockage_rate': float(blockage_info['blockage_rate']),
                    'blockage_count': int(blockage_info['blockage_count']),
                    'blockage_severity': float(min(1.0, blockage_info['severity'])),
                    'total_foreground_voxels': blockage_info['total_foreground_voxels'],
                    'blocked_voxels': blockage_info['blocked_voxels']
                }
                all_results.append(result)

                # Save visualization: MRI + segmentation + blockage overlay
                _save_blockage_visualization(
                    img_np, pred_np, blockage_mask, blockage_info,
                    model_name, sample_count + 1
                )
                print(f"    Sample {sample_count + 1}: blockage_rate={blockage_info['blockage_rate']:.2f}%, "
                      f"count={blockage_info['blockage_count']}, severity={blockage_info['severity']:.3f}")

                sample_count += 1

    # Save results
    out_json = os.path.join(_script_dir, 'blockage_detection_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved {out_json}")

    df = pd.DataFrame(all_results)
    out_csv = os.path.join(_script_dir, 'blockage_detection_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"✓ Saved {out_csv}")

    # Summary bar chart: mean blockage rate / count / severity per model (5 samples each)
    _save_blockage_comparison_chart(all_results)
    print(f"✓ Saved blockage_detection_comparison.png")

    print("\n" + "="*60)
    print("BLOCKAGE DETECTION DEMO COMPLETE (3 models × 5 samples)")
    print("="*60)
    print("Generated: blockage_detection_<model>_sample<N>.png (15 images)")
    print("           blockage_detection_results.json, .csv")
    print("           blockage_detection_comparison.png")


def _save_blockage_visualization(img_np, pred_np, blockage_mask, blockage_info, model_name, sample_num):
    """Save one figure: MRI, segmentation, blockage overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    mid_z = img_np.shape[2] // 2

    axes[0].imshow(img_np[:, :, mid_z], cmap='gray')
    axes[0].set_title('MRI (mid slice)', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_np[:, :, mid_z], cmap='gray', alpha=0.6)
    axes[1].imshow(pred_np[:, :, mid_z], cmap='Blues', alpha=0.6)
    axes[1].set_title('Segmentation', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img_np[:, :, mid_z], cmap='gray', alpha=0.6)
    axes[2].imshow(pred_np[:, :, mid_z], cmap='Blues', alpha=0.3)
    axes[2].imshow(blockage_mask[:, :, mid_z], cmap='Reds', alpha=0.8)
    axes[2].set_title(f'Blockage Detection\nRate: {blockage_info["blockage_rate"]:.2f}% | '
                     f'Count: {blockage_info["blockage_count"]} | Severity: {blockage_info["severity"]:.3f}',
                     fontsize=11, fontweight='bold')
    axes[2].axis('off')

    safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
    plt.suptitle(f'{model_name} — Blockage Detection Sample {sample_num}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(_script_dir, f'blockage_detection_{safe_name}_sample{sample_num}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def _save_blockage_comparison_chart(all_results):
    """Bar chart: mean blockage rate, mean count, mean severity per model (5 samples each)."""
    df = pd.DataFrame(all_results)
    summary = df.groupby('model_name').agg({
        'blockage_rate': ['mean', 'std'],
        'blockage_count': ['mean', 'std'],
        'blockage_severity': ['mean', 'std']
    }).reset_index()
    summary.columns = ['model_name', 'rate_mean', 'rate_std', 'count_mean', 'count_std', 'severity_mean', 'severity_std']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    models = summary['model_name'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    axes[0].bar(models, summary['rate_mean'], yerr=summary['rate_std'], capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Blockage Rate (%)')
    axes[0].set_title('Mean Blockage Rate (5 samples/model)', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (m, v) in enumerate(zip(summary['rate_mean'], summary['rate_std'])):
        axes[0].text(i, m + v + 1, f'{m:.1f}%', ha='center', fontweight='bold')

    axes[1].bar(models, summary['count_mean'], yerr=summary['count_std'], capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Blockage Count')
    axes[1].set_title('Mean Blockage Count (5 samples/model)', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(summary['count_mean']):
        axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')

    axes[2].bar(models, summary['severity_mean'], yerr=summary['severity_std'], capsize=5, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Severity (0–1)')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title('Mean Severity (5 samples/model)', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, (m, v) in enumerate(zip(summary['severity_mean'], summary['severity_std'])):
        axes[2].text(i, min(m + v + 0.02, 1.0), f'{m:.3f}', ha='center', fontweight='bold')

    plt.suptitle('Blockage Detection: 3 Models × 5 Samples Each', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(_script_dir, 'blockage_detection_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print("="*60)
    print("BLOCKAGE DETECTION DEMO — 3 models, 5 samples per model")
    print("="*60)
    run_blockage_detection_demo()
