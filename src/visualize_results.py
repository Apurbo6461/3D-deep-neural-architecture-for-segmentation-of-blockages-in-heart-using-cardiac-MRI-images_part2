"""
Comprehensive visualization script for model comparison and 3D blockage detection visualization.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import json
from torch.utils.data import DataLoader

from data.dataset import MedicalDataset
from models.unet3d_standard import UNet3D
from models.vnet import VNet3D
from models.res_att_unet3d import ResAttUNet3D
from blockage_detection import BlockageDetector
from utils import calculate_dice_score, calculate_iou, calculate_accuracy

def load_evaluation_results():
    """Load evaluation results if available, otherwise return None."""
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r") as f:
            return json.load(f)
    elif os.path.exists("evaluation_results.csv"):
        df = pd.read_csv("evaluation_results.csv")
        return df.to_dict('records')
    return None

def create_accuracy_comparison_bar_chart(results=None):
    """Create bar chart comparing accuracy metrics across models."""
    if results is None:
        results = load_evaluation_results()
    
    if results is None:
        print("No evaluation results found. Please run evaluate_models.py first.")
        return
    
    model_names = [r['model_name'] for r in results]
    
    # Extract metrics
    dice_scores = [r['dice_score'] for r in results]
    iou_scores = [r['iou_score'] for r in results]
    acc_scores = [r['accuracy'] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(model_names))
    width = 0.6
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Dice Score
    bars1 = axes[0].bar(x, dice_scores, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Dice Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, dice_scores)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # IoU Score
    bars2 = axes[1].bar(x, iou_scores, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    axes[1].set_title('IoU Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_axisbelow(True)
    
    for i, (bar, score) in enumerate(zip(bars2, iou_scores)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Accuracy
    bars3 = axes[2].bar(x, acc_scores, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[2].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[2].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_names, rotation=15, ha='right')
    axes[2].set_ylim([0, 1.0])
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_axisbelow(True)
    
    for i, (bar, score) in enumerate(zip(bars3, acc_scores)):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Saved accuracy comparison to accuracy_comparison.png")
    plt.close()

def create_blockage_rate_comparison_bar_chart(results=None):
    """Create bar chart comparing blockage detection rates across models."""
    if results is None:
        results = load_evaluation_results()
    
    if results is None:
        print("No evaluation results found. Please run evaluate_models.py first.")
        return
    
    model_names = [r['model_name'] for r in results]
    
    # Extract blockage metrics
    blockage_rates = [r['mean_blockage_rate'] for r in results]
    blockage_detection_acc = [r['blockage_detection_accuracy'] for r in results]
    blockage_f1 = [r['blockage_detection_f1'] for r in results]
    blockage_precision = [r['blockage_detection_precision'] for r in results]
    blockage_recall = [r['blockage_detection_recall'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x = np.arange(len(model_names))
    width = 0.6
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Blockage Rate (%)
    bars1 = axes[0, 0].bar(x, blockage_rates, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Blockage Rate (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Mean Blockage Rate Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_axisbelow(True)
    
    for i, (bar, rate) in enumerate(zip(bars1, blockage_rates)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Blockage Detection Accuracy
    bars2 = axes[0, 1].bar(x, blockage_detection_acc, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Detection Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Blockage Detection Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_axisbelow(True)
    
    for i, (bar, acc) in enumerate(zip(bars2, blockage_detection_acc)):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Blockage Detection F1
    bars3 = axes[1, 0].bar(x, blockage_f1, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Blockage Detection F1 Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_axisbelow(True)
    
    for i, (bar, f1) in enumerate(zip(bars3, blockage_f1)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Precision and Recall
    x_pos = np.arange(len(model_names))
    width_bar = 0.35
    
    bars4a = axes[1, 1].bar(x_pos - width_bar/2, blockage_precision, width_bar, 
                           label='Precision', color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4b = axes[1, 1].bar(x_pos + width_bar/2, blockage_recall, width_bar,
                           label='Recall', color='#9467bd', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    axes[1, 1].set_xlabel('Model', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Blockage Detection Precision vs Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_axisbelow(True)
    
    for bar, prec in zip(bars4a, blockage_precision):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{prec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    for bar, rec in zip(bars4b, blockage_recall):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{rec:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("blockage_rate_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Saved blockage rate comparison to blockage_rate_comparison.png")
    plt.close()

def visualize_3d_blockage_detection(model, model_name, device, num_samples=3):
    """Create 3D visualizations showing original image, segmentation, and detected blockages."""
    TEST_DIR = "/Users/abidhasan/Downloads/for thesis/dataset copy 6/testing"
    TARGET_SHAPE = (128, 128, 64)
    
    # Load test dataset
    test_dataset = MedicalDataset(root_dir=TEST_DIR, target_shape=TARGET_SHAPE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model weights if available
    # Fixing filename format logic to match training/evaluation
    # Training saves as: best_{model_name.lower().replace(' ', '-')}.pth (e.g. best_resatt-3d-u-net.pth)
    model_path = f"best_{model_name.lower().replace(' ', '-')}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    
    model = model.to(device)
    model.eval()
    
    detector = BlockageDetector(threshold_thickening=0.3, min_blockage_size=5)
    
    sample_count = 0
    
    with torch.no_grad():
        for imgs, gts in test_loader:
            if sample_count >= num_samples:
                break
            
            imgs = imgs.to(device)
            gts = gts.cpu()
            
            # Get prediction
            preds = model(imgs)
            if preds.shape[1] == 1:
                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = (preds_sigmoid > 0.5).float()
            else:
                preds_sigmoid = torch.softmax(preds, dim=1)
                preds_binary = (preds_sigmoid[:, 0:1] < 0.5).float()
            
            # Convert to numpy
            img_np = imgs[0, 0].cpu().numpy()
            pred_np = preds_binary[0, 0].cpu().numpy()
            gt_np = gts[0, 0].numpy()
            
            # Detect blockages
            blockage_info = detector.detect_blockages(pred_np, img_np)
            blockage_mask = blockage_info['blockage_mask']
            
            # Create 3D visualization
            fig = plt.figure(figsize=(20, 6))
            
            # Get middle slices for visualization
            mid_z = img_np.shape[2] // 2
            mid_y = img_np.shape[1] // 2
            mid_x = img_np.shape[0] // 2
            
            # Original Image - XY slice
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(img_np[:, :, mid_z], cmap='gray')
            ax1.set_title('Original Image (XY slice)', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Segmentation - XY slice
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(img_np[:, :, mid_z], cmap='gray', alpha=0.5)
            ax2.imshow(pred_np[:, :, mid_z], cmap='Reds', alpha=0.5)
            ax2.set_title(f'Segmentation (Dice: {calculate_dice_score(preds_binary[0:1], gts[0:1].to(device)):.3f})', 
                         fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            # Blockage Detection - XY slice
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow(img_np[:, :, mid_z], cmap='gray', alpha=0.7)
            ax3.imshow(pred_np[:, :, mid_z], cmap='Blues', alpha=0.3)
            ax3.imshow(blockage_mask[:, :, mid_z], cmap='Reds', alpha=0.6)
            ax3.set_title(f'Blockage Detection\nRate: {blockage_info["blockage_rate"]:.2f}%', 
                         fontsize=12, fontweight='bold')
            ax3.axis('off')
            
            # 3D Volume Visualization
            ax4 = fig.add_subplot(1, 4, 4, projection='3d')
            
            # Sample points for 3D visualization (downsample for performance)
            step = 4
            x, y, z = np.meshgrid(
                np.arange(0, img_np.shape[0], step),
                np.arange(0, img_np.shape[1], step),
                np.arange(0, img_np.shape[2], step)
            )
            
            # Get values at sampled points
            pred_sampled = pred_np[::step, ::step, ::step]
            blockage_sampled = blockage_mask[::step, ::step, ::step]
            
            # Plot segmentation
            seg_points = pred_sampled > 0.5
            if seg_points.any():
                ax4.scatter(x[seg_points], y[seg_points], z[seg_points], 
                           c='blue', alpha=0.1, s=1, label='Segmentation')
            
            # Plot blockages
            blockage_points = blockage_sampled > 0.5
            if blockage_points.any():
                ax4.scatter(x[blockage_points], y[blockage_points], z[blockage_points],
                           c='red', alpha=0.8, s=5, label='Blockages')
            
            ax4.set_title('3D Volume View', fontsize=12, fontweight='bold')
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.legend()
            
            plt.suptitle(f'{model_name} - Sample {sample_count + 1}\n'
                        f'Blockage Rate: {blockage_info["blockage_rate"]:.2f}% | '
                        f'Count: {blockage_info["blockage_count"]} | '
                        f'Severity: {blockage_info["severity"]:.3f}',
                        fontsize=14, fontweight='bold', y=1.02)
            
            plt.tight_layout()
            filename = f"3d_blockage_detection_{model_name.lower().replace(' ', '_').replace('-', '_')}_sample{sample_count+1}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 3D visualization to {filename}")
            plt.close()
            
            sample_count += 1

def main():
    """Main function to create all visualizations."""
    print("="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Check if evaluation results exist
    results = load_evaluation_results()
    
    if results is None:
        print("\n⚠ No evaluation results found.")
        print("Running evaluation first...")
        print("="*60)
        
        # Import and run evaluation
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from evaluate_models import main as eval_main
        eval_main()
        
        # Reload results
        results = load_evaluation_results()
    
    if results:
        print("\n" + "="*60)
        print("CREATING ACCURACY COMPARISON CHARTS")
        print("="*60)
        create_accuracy_comparison_bar_chart(results)
        
        print("\n" + "="*60)
        print("CREATING BLOCKAGE RATE COMPARISON CHARTS")
        print("="*60)
        create_blockage_rate_comparison_bar_chart(results)
    
    # Create 3D visualizations
    print("\n" + "="*60)
    print("CREATING 3D BLOCKAGE DETECTION VISUALIZATIONS")
    print("="*60)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu" and torch.backends.mps.is_available():
        DEVICE = "mps"
    
    models_config = {
        '3D U-Net': UNet3D(in_channels=1, out_channels=1, base_filters=32),
        'V-Net': VNet3D(in_channels=1, out_channels=1, base_filters=16),
        'ResAtt-3D-U-Net': ResAttUNet3D(in_channels=1, out_channels=1, base_filters=16)
    }
    
    for model_name, model in models_config.items():
        print(f"\nProcessing {model_name}...")
        try:
            visualize_3d_blockage_detection(model, model_name, DEVICE, num_samples=2)
        except Exception as e:
            print(f"⚠ Error visualizing {model_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - accuracy_comparison.png")
    print("  - blockage_rate_comparison.png")
    print("  - 3d_blockage_detection_*.png (for each model)")

if __name__ == "__main__":
    main()
