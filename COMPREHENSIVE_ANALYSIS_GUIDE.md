# Comprehensive Blockage Analysis Guide

## Overview

This guide explains how to use the comprehensive blockage detection and segmentation pipeline for your thesis: **"An efficient 3D deep neural architecture for segmentation of blockages in heart using cardiac MRI images"**.

## What the Pipeline Does

The comprehensive analysis pipeline performs:

1. **3D Segmentation**: Uses deep neural networks (3D U-Net, V-Net, ResAtt-3D-U-Net) to segment cardiac structures from MRI images
2. **Blockage Detection**: Detects blockages in the segmented cardiac structures using multiple methods:
   - Vessel narrowing detection
   - Disconnected region analysis
   - Intensity-based anomaly detection
3. **Anatomical Region Identification**: Identifies which part of the heart contains blockages:
   - **LV** (Left Ventricle)
   - **RV** (Right Ventricle)
   - **Myocardial** (Myocardium)
   - **Artery** (Coronary Arteries)
4. **Comprehensive Reporting**: Generates:
   - Accuracy metrics and comparison graphs
   - Blockage detection statistics
   - Anatomical region analysis
   - Individual sample visualizations

## Files Created

### Main Script
- `src/comprehensive_blockage_analysis.py` - Main comprehensive analysis pipeline

### Supporting Modules
- `src/anatomical_region_identifier.py` - Identifies anatomical regions (LV, RV, Myocardial, Arteries)
- `src/blockage_detection.py` - Detects blockages in segmented structures

### Output Files Generated
- `comprehensive_analysis_results.json` - Detailed results in JSON format
- `comprehensive_analysis_results.csv` - Results in CSV format for analysis
- `accuracy_comparison_comprehensive.png` - Comprehensive accuracy comparison graph
- `blockage_region_analysis.png` - Analysis of blockages by anatomical region
- `individual_sample_*.png` - Individual sample visualizations (first 5 samples)

## Usage

### Full Analysis (All Test Samples)

```bash
cd src
python3 comprehensive_blockage_analysis.py
```

This will:
- Process all test samples
- Generate comprehensive reports
- Create all visualizations
- Save results in multiple formats

### Quick Test (Limited Samples)

For testing, use the quick test script:

```bash
cd src
python3 quick_comprehensive_test.py
```

This processes only 3 samples for quick verification.

### Custom Analysis

You can also import and use the pipeline programmatically:

```python
from comprehensive_blockage_analysis import ComprehensiveBlockageAnalysis

# Initialize
analyzer = ComprehensiveBlockageAnalysis(
    test_dir="/path/to/testing",
    device='auto',  # or 'cuda', 'mps', 'cpu'
    target_shape=(128, 128, 64)
)

# Run analysis
analyzer.analyze_dataset(
    num_samples=10,  # None for all samples
    save_individual=True  # Save individual visualizations
)
```

## Output Interpretation

### Accuracy Comparison Graph

The `accuracy_comparison_comprehensive.png` shows:
- **Dice Score**: Segmentation overlap accuracy
- **IoU Score**: Intersection over Union
- **Accuracy**: Pixel-wise accuracy
- **Sensitivity**: True positive rate
- **Specificity**: True negative rate
- **Blockage Rate**: Mean blockage percentage

### Blockage Region Analysis

The `blockage_region_analysis.png` shows:
- Distribution of blockages across anatomical regions
- Histograms of blockage rates for each region
- Frequency of blockages in LV, RV, Myocardial, and Arteries

### Individual Sample Visualizations

Each sample visualization shows:
1. Original MRI image
2. Ground truth segmentation
3. Predicted segmentation with Dice score
4. Blockage detection overlay
5. Anatomical regions (LV, RV, Myocardial, Artery) with blockage highlights
6. Metrics summary
7. Blockage distribution by region

## Key Features

### Anatomical Region Identification

The pipeline automatically identifies which part of the heart contains blockages:

- **Left Ventricle (LV)**: Large chamber, typically left-central position
- **Right Ventricle (RV)**: Large chamber, typically right-anterior position
- **Myocardium**: Medium-sized region surrounding LV
- **Coronary Arteries**: Small, elongated tubular structures

The identification uses:
- Spatial location analysis
- Morphological features (compactness, elongation)
- Size-based classification
- Distance transform features

### Blockage Detection Methods

1. **Vessel Narrowing**: Uses distance transform to detect narrowed regions
2. **Disconnected Regions**: Identifies isolated regions that may indicate blockages
3. **Intensity Anomalies**: Detects intensity variations that may indicate blockages

## Requirements

All dependencies are listed in `requirements.txt`:
- torch >= 2.0.0
- nibabel >= 5.0.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- tqdm >= 4.60.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

## Model Weights

The pipeline expects model weights in the `src/` directory:
- `best_3d_u-net.pth` - 3D U-Net weights
- `best_v-net.pth` - V-Net weights
- `best_resatt_3d_u_net.pth` - ResAtt-3D-U-Net weights (optional)

If weights are missing, the pipeline will use random initialization (for testing only).

## Thesis Justification

This pipeline provides comprehensive justification for your thesis title by:

1. **3D Deep Neural Architecture**: Uses multiple 3D architectures (U-Net, V-Net, ResAtt-3D-U-Net)
2. **Segmentation**: Performs accurate segmentation of cardiac structures
3. **Blockage Detection**: Specifically detects blockages in segmented structures
4. **Anatomical Localization**: Identifies which part of the heart (LV, RV, Myocardial, Arteries) contains blockages
5. **Comprehensive Evaluation**: Provides accuracy metrics, graphs, and visualizations

## Troubleshooting

### Out of Memory
- Reduce `target_shape` to smaller dimensions
- Process fewer samples at a time
- Use CPU instead of GPU/MPS

### Model Weights Not Found
- Train models first using `train_models.py`
- Or use the pipeline with random initialization for testing

### Slow Processing
- Use `quick_comprehensive_test.py` for testing
- Reduce `num_samples` parameter
- Use GPU/MPS if available

## Citation

When using this pipeline for your thesis, please cite:
- The cardiac MRI dataset used
- The deep learning frameworks (PyTorch)
- Relevant papers for the architectures (U-Net, V-Net, etc.)

## Support

For issues or questions, check:
- `README.md` - General project information
- `SEGMENTATION_GUIDE.md` - Segmentation details
- `THESIS_JUSTIFICATION.md` - Thesis justification document
