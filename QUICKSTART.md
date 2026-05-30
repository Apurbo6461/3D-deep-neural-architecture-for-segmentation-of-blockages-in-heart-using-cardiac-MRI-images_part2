# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train Models

Navigate to the src directory and run:

```bash
cd src
python train_models.py
```

This will:
- Train 3D U-Net, V-Net, and ResAtt-3D-U-Net models
- Save best model weights (best_*.pth files)
- Generate training logs and comparison plots
- Take approximately 1-2 hours depending on your hardware

## Step 3: Evaluate Models

After training, evaluate on the test set:

```bash
python evaluate_models.py
```

This will:
- Load trained models
- Evaluate on test dataset
- Calculate segmentation metrics (Dice, IoU, Accuracy, etc.)
- Calculate blockage detection metrics
- Generate comparison visualizations

## Step 4: View Results

Check the generated files:

1. **evaluation_results.csv** - Tabular comparison of all models
2. **evaluation_results.json** - Detailed JSON results
3. **model_comparison.png** - Visual comparison charts
4. **training_comparison.png** - Training curves for all models

## Expected Results

You should see metrics like:

- **Dice Score**: 0.80-0.90 (higher is better)
- **IoU Score**: 0.70-0.85 (higher is better)
- **Accuracy**: 0.90-0.98 (higher is better)
- **Blockage Detection Accuracy**: 0.70-0.90 (higher is better)
- **Blockage Detection F1**: 0.70-0.90 (higher is better)

## All-in-One Command

To train and evaluate in one go:

```bash
cd src
python main.py --mode both --epochs 20
```

## Troubleshooting

**Out of Memory Error:**
- Reduce TARGET_SHAPE in dataset.py (e.g., from (128, 128, 64) to (96, 96, 48))
- Ensure batch_size is 1

**Model Weights Not Found:**
- Make sure you've run training first
- Check that best_*.pth files exist in the root directory

**Import Errors:**
- Make sure you're in the `src` directory
- Or add src to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
