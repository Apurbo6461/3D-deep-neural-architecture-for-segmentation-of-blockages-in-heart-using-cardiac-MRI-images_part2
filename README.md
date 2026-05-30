# Heart Blockage Detection using 3D Deep Learning Models

This project implements and compares multiple 3D deep learning models for heart blockage detection from medical imaging data (ACDC dataset).

## Models Implemented

1. **3D U-Net**: Standard 3D U-Net architecture with encoder-decoder structure
2. **V-Net**: 3D V-Net with residual connections and instance normalization
3. **ResAtt-3D-U-Net**: Residual Attention 3D U-Net with attention gates

## Features

- **3D Medical Image Segmentation**: Handles NIfTI format medical images
- **Blockage Detection**: Advanced algorithm to detect heart blockages from segmented regions
- **Model Comparison**: Comprehensive evaluation comparing all three models
- **Metrics**: Dice score, IoU, accuracy, sensitivity, specificity, and blockage detection rates

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── training/
│   ├── patient001/
│   │   ├── patient001_frame01.nii
│   │   ├── patient001_frame01_gt.nii
│   │   └── ...
│   └── ...
└── testing/
    ├── patient101/
    │   ├── patient101_frame01.nii
    │   ├── patient101_frame01_gt.nii
    │   └── ...
    └── ...
```

## Usage

### Training Models

Train all three models:
```bash
cd src
python train_models.py
```

Or use the main script:
```bash
cd src
python main.py --mode train --epochs 20
```

### Evaluating Models

Evaluate trained models on test set:
```bash
cd src
python evaluate_models.py
```

Or use the main script:
```bash
cd src
python main.py --mode evaluate
```

### Training and Evaluation

Run both training and evaluation:
```bash
cd src
python main.py --mode both --epochs 20
```

## Output Files

After training and evaluation, you'll get:

1. **Model Weights**:
   - `best_3d_u_net.pth`
   - `best_v_net.pth`
   - `best_resatt_3d_u_net.pth`

2. **Training Logs**:
   - `training_log_3d_u_net.csv`
   - `training_log_v_net.csv`
   - `training_log_resatt_3d_u_net.csv`
   - `training_comparison.png`

3. **Evaluation Results**:
   - `evaluation_results.csv`: Tabular results
   - `evaluation_results.json`: Detailed JSON results
   - `model_comparison.png`: Visualization comparing all models

## Metrics Explained

### Segmentation Metrics
- **Dice Score**: Measures overlap between prediction and ground truth (0-1, higher is better)
- **IoU (Intersection over Union)**: Similar to Dice, measures overlap
- **Accuracy**: Pixel-wise classification accuracy
- **Sensitivity**: True positive rate (recall)
- **Specificity**: True negative rate

### Blockage Detection Metrics
- **Blockage Detection Accuracy**: Accuracy of detecting blockages
- **Blockage Detection Precision**: Precision of blockage detection
- **Blockage Detection Recall**: Recall of blockage detection
- **Blockage Detection F1**: F1 score for blockage detection
- **Mean Blockage Rate**: Average percentage of blocked area
- **Mean Severity**: Average severity score (0-1)

## Blockage Detection Algorithm

The blockage detection algorithm uses multiple methods:

1. **Vessel Narrowing Detection**: Uses distance transform to identify narrow regions
2. **Disconnected Regions**: Detects isolated regions that might indicate blockages
3. **Intensity Anomalies**: Identifies regions with abnormal intensity patterns

These methods are combined to create a comprehensive blockage detection system.

## Model Architecture Details

### 3D U-Net
- Standard encoder-decoder architecture
- 4 levels of downsampling/upsampling
- Skip connections between encoder and decoder
- Base filters: 32

### V-Net
- Residual blocks with instance normalization
- PReLU activation
- 5x5x5 convolutions
- Base filters: 16

### ResAtt-3D-U-Net
- Residual blocks in encoder/decoder
- Attention gates for skip connections
- Batch normalization
- Base filters: 16

## Citation

If you use this code with the ACDC dataset, please cite:

```
O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" 
in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502
```

## Notes

- 3D models are memory-intensive. Batch size is set to 1 by default.
- Training time depends on GPU availability. MPS (Metal Performance Shaders) is used on Mac M1/M2/M3.
- The models are designed for binary segmentation (foreground vs background).
- For multi-class segmentation, modify the output channels and loss functions accordingly.

## Troubleshooting

1. **Out of Memory**: Reduce batch size or target shape in dataset.py
2. **Slow Training**: Use GPU if available, or reduce number of epochs
3. **Import Errors**: Make sure you're in the `src` directory or adjust Python path
