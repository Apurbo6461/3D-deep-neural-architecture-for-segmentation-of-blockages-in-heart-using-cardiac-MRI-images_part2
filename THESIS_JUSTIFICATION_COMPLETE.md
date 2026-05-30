# Complete Thesis Justification Document

## Thesis Title
**"An efficient 3D deep neural architecture for segmentation of blockages in heart using cardiac MRI images"**

---

## How This Thesis Title is Justified

Each part of the title is directly supported by the implementation, metrics, and charts produced in this project:

| Title phrase | Justification | Evidence |
|--------------|---------------|----------|
| **Efficient** | Three 3D architectures are compared on **efficiency** (model size, inference time, throughput, memory). ResAtt-3D-U-Net offers a favourable accuracy–efficiency trade-off. | **Efficiency chart**: `src/efficiency_comparison.png` — model size (MB), parameters (M), inference time (ms), throughput (samples/s). Data: `src/efficiency_analysis_results.json`. |
| **3D** | All models use **3D** convolutions and process **3D** volumes (e.g. 128×128×64). No 2D slices; full volumetric context is preserved. | Architectures in `src/models/`: `unet3d_standard.py`, `vnet.py`, `unet3d.py`. Input/output are 3D tensors. |
| **Deep neural architecture** | **Deep** encoder–decoder networks with multiple layers (3D U-Net, V-Net, ResAtt-3D-U-Net). | Three distinct deep architectures; millions of parameters each (e.g. 22.6M, 7.6M, 4.2M). |
| **Segmentation** | The task is **segmentation** of cardiac structures from MRI. Performance is measured by Dice, IoU, accuracy, sensitivity, specificity. | **Segmentation / accuracy bar charts**: `src/model_comparison.png`, `src/accuracy_comparison.png`, `src/accuracy_comparison_comprehensive.png`. Data: `src/evaluation_results.json`, `src/comprehensive_analysis_results.json`. |
| **Blockages** | **Blockages** are detected inside the segmented structures (narrowing, disconnected regions, intensity anomalies). Blockage rate, count, and severity are reported. | **Blockage detection charts**: `src/blockage_rate_comparison.png`, `src/blockage_detection_comparison.png`, `src/blockage_region_analysis.png`. Data: `src/blockage_detection_results.json`, blockage metrics in `evaluation_results.json`. |
| **in heart** | Blockages are localized to **heart** regions: Left Ventricle (LV), Right Ventricle (RV), Myocardium. | Anatomical region identifier: `src/anatomical_region_identifier.py`. Visualizations: `src/blockage_region_analysis.png`, `src/individual_sample_*.png`. |
| **using cardiac MRI images** | Inputs are **cardiac MRI** volumes in NIfTI format; preprocessing and evaluation use real cardiac MRI data. | Dataset: `src/data/dataset.py`; test/training NIfTI (`.nii`) from cardiac MRI. Paths in scripts point to cardiac MRI directories. |

**Summary:** The title is justified because (1) **efficiency** is analysed and reported with dedicated metrics and charts; (2) **3D** architectures and volumes are used throughout; (3) **deep** 3D networks are implemented and compared; (4) **segmentation** is the main task, with standard metrics and bar charts; (5) **blockages** are detected and visualised with dedicated charts; (6) **heart** regions (LV, RV, Myocardium) are identified; (7) **cardiac MRI images** are the only input modality.

---

## Four Charts Quick Reference

| Chart you need | File(s) in `src/` |
|----------------|--------------------|
| **Accuracy bar chart** | `accuracy_comparison.png`, `accuracy_comparison_comprehensive.png` |
| **Blockage detection chart** | `blockage_rate_comparison.png`, `blockage_detection_comparison.png` |
| **Segmentation chart** | `model_comparison.png` (Dice, IoU, Accuracy, Sensitivity, Specificity, Blockage metrics), `accuracy_comparison_comprehensive.png`, `blockage_region_analysis.png` |
| **Efficiency chart** | `efficiency_comparison.png` |

---

## Updated Charts for the Thesis (All in `src/`)

Generate or refresh them by running from `src/`:

1. **Accuracy bar chart**  
   - **Files:** `accuracy_comparison.png`, `accuracy_comparison_comprehensive.png`  
   - **Content:** Dice, IoU, accuracy (and optionally sensitivity, specificity) per model.  
   - **How to update:** Run `python3 evaluate_models.py` (20 samples), then `python3 quick_run.py`. Comprehensive analysis: `python3 comprehensive_blockage_analysis.py` (20 samples).

2. **Blockage detection chart**  
   - **Files:** `blockage_rate_comparison.png`, `blockage_detection_comparison.png`  
   - **Content:** Blockage rate (%), blockage count, severity; comparison across models (and optionally by heart region).  
   - **How to update:** Same evaluation + `quick_run` for blockage rate; `python3 blockage_detection_demo.py` for blockage detection comparison (3 models × 5 samples).

3. **Segmentation chart**  
   - **Files:** `model_comparison.png`, `accuracy_comparison_comprehensive.png`, `blockage_region_analysis.png`  
   - **Content:** Segmentation metrics (Dice, IoU, accuracy, sensitivity, specificity) and, where applicable, blockage by region.  
   - **How to update:** `python3 evaluate_models.py` produces `model_comparison.png`; comprehensive analysis produces `accuracy_comparison_comprehensive.png` and `blockage_region_analysis.png`.

4. **Efficiency chart**  
   - **Files:** `efficiency_comparison.png`  
   - **Content:** Model size (MB), parameter count (M), inference time (ms), throughput (samples/s), memory.  
   - **How to update:** `python3 efficiency_analysis.py`.

**One-command refresh (from `src/`):**  
`python3 run_all.py` — runs comprehensive analysis (20 samples), evaluation (20 samples), bar charts, 3D visualizations, and blockage detection demo. Then run `python3 efficiency_analysis.py` if the efficiency chart is not yet updated.

---

## Detailed Justification by Component

### 1. "3D Deep Neural Architecture"

- **3D U-Net** (`unet3d_standard.py`): 4-level 3D U-Net, encoder–decoder, 32 base filters.  
- **V-Net** (`vnet.py`): 3D V-Net with residual connections.  
- **ResAtt-3D-U-Net** (`unet3d.py`): 3D U-Net with residual and attention blocks.  

All use 3D convolutions and 3D volumes (e.g. 128×128×64).  
✅ **FULFILLED.**

---

### 2. "Segmentation of Blockages"

- **Segmentation:** Models output binary (or multi-class) segmentation; metrics include Dice, IoU, accuracy, sensitivity, specificity (all in [0, 1]).  
- **Blockage detection** (`blockage_detection.py`): Vessel narrowing (distance transform), disconnected regions, intensity anomalies; outputs blockage rate (%), count, severity.  

Evidence: `evaluation_results.json`, `comprehensive_analysis_results.json`, blockage detection results and visualisations.  
✅ **FULFILLED.**

---

### 3. "in Heart"

- **Anatomical regions** (`anatomical_region_identifier.py`): LV, RV, Myocardium (no artery in the current pipeline). Blockages are attributed to these heart regions.  

Evidence: `blockage_region_analysis.png`, region-wise blockage in individual sample figures.  
✅ **FULFILLED.**

---

### 4. "using Cardiac MRI Images"

- Input: NIfTI cardiac MRI volumes; dataset loader in `data/dataset.py`; preprocessing (resize, normalize) for 3D cardiac MRI.  

Evidence: Use of cardiac MRI paths and NIfTI loading in code and config.  
✅ **FULFILLED.**

---

### 5. "Efficient"

- **Efficiency analysis** (`efficiency_analysis.py`): Model size (MB), parameter count (millions), inference time (ms), throughput (samples/s), CPU/GPU memory. Comparison across the three architectures.  

Evidence: `efficiency_comparison.png`, `efficiency_analysis_results.json`.  
✅ **FULFILLED.**

---

## Output Files Summary

| Purpose | Files (in `src/`) |
|--------|--------------------|
| **Accuracy bar chart** | `accuracy_comparison.png`, `accuracy_comparison_comprehensive.png` |
| **Blockage detection chart** | `blockage_rate_comparison.png`, `blockage_detection_comparison.png` |
| **Segmentation chart** | `model_comparison.png`, `accuracy_comparison_comprehensive.png`, `blockage_region_analysis.png` |
| **Efficiency chart** | `efficiency_comparison.png` |
| **Data** | `evaluation_results.json`, `comprehensive_analysis_results.json`, `blockage_detection_results.json`, `efficiency_analysis_results.json` (and corresponding `.csv`) |

---

## Thesis Title Justification Checklist

| Component | Status | Evidence |
|-----------|--------|----------|
| **Efficient** | ✅ | Efficiency metrics and `efficiency_comparison.png` |
| **3D** | ✅ | 3D convolutions and volumes in all models |
| **Deep neural architecture** | ✅ | Three deep 3D architectures (U-Net, V-Net, ResAtt-3D-U-Net) |
| **Segmentation** | ✅ | Dice, IoU, accuracy; segmentation bar charts |
| **Blockages** | ✅ | Blockage detection module and blockage charts |
| **in heart** | ✅ | LV, RV, Myocardium region identification |
| **Cardiac MRI images** | ✅ | NIfTI cardiac MRI as input |

---

## Conclusion

The thesis title **"An efficient 3D deep neural architecture for segmentation of blockages in heart using cardiac MRI images"** is justified by:

1. **Efficient** — Quantified via model size, inference time, throughput, and memory; reported in the **efficiency chart** (`efficiency_comparison.png`).  
2. **3D** — All models and data are 3D volumetric.  
3. **Deep neural architecture** — Three 3D deep networks are implemented and compared.  
4. **Segmentation** — Primary task with standard metrics; shown in **accuracy/segmentation bar charts**.  
5. **Blockages** — Detected and compared across models; shown in **blockage detection charts**.  
6. **in heart** — Blockages localised to LV, RV, and Myocardium.  
7. **Cardiac MRI images** — Only input modality; NIfTI cardiac MRI used throughout.

Including the four chart types — **accuracy bar chart**, **blockage detection chart**, **segmentation chart**, and **efficiency chart** — in the thesis will directly support this justification.
