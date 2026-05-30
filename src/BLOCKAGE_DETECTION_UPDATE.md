# Multi-Metric Blockage Detection Update

In accordance with the scientific framework provided for cardiac MRI, the blockage detection methodology has been completely overhauled from a simple static "narrowing" approach to a robust, multi-metric functional analysis using paired End-Diastole (ED) and End-Systole (ES) frames.

## 1. Pipeline Overhaul (ED vs ES)
Instead of analyzing individual, independent 3D slices, the `ComprehensiveBlockageAnalysis` pipeline now loops through patients. For each patient, it explicitly extracts the **ED frame** and the **ES frame** (identified via the dataset's `Info.cfg`). 

Both frames are passed through the deep learning models to predict their respective cardiac segmentations. The paired segmentations are then analyzed jointly to determine functional metrics.

## 2. Multi-Metric Framework Implemented
The `BlockageDetector` class was completely rewritten to implement the following formulas:

### Primary Metric: Regional Wall Thickening
Blockages (ischemic/infarcted regions) reduce the myocardium's ability to contract.
- **Formula:** `(T_es - T_ed) / T_ed * 100` (computed pointwise using 3D distance transforms on the myocardium masks).
- **Threshold:** Regions showing **< 35%** thickening are explicitly flagged as potential blockages. This provides a direct, highly local map of where the blockages are located.

### Secondary Metric: LV Cavity Deformation
- **Formula:** `(EDV_voxels - ESV_voxels) / EDV_voxels * 100`
- Measures how much the ventricular cavity shrinks during a heartbeat, acting as a proxy for regional asymmetry and poor cavity contraction.

### Global Validation: Ejection Fraction (EF)
- Acts as a clinical validation metric, derived mathematically from the cavity deformation, to check for severe global dysfunction (e.g., EF < 40%).

### Structural Check: Wall Thickness
- A baseline measurement of ED myocardial thickness to flag severe thinning (indicative of chronic scar/infarction) or hypertrophy.

## 3. Combined Abnormality Score
A new scoring system was implemented to provide a defensible "Blockage Likelihood" rather than a hard threshold:
- **0 Points:** Normal
- **1-2 Points:** Low/Moderate Suspicion
- **3+ Points:** High Suspicion (Severe thickening impairment + cavity abnormality + low EF)

## 4. Enhanced Visualizations
The output plots (`individual_sample_...png`) have been updated to seamlessly integrate this new framework:
- The exact regions where **Regional Wall Thickening is < 35%** are now mapped back onto the original MRI and highlighted in a 'hot' colormap to "show perfectly on the plots where the blockages are."
- The metrics summary panel now prints the full suite of dysfunction parameters: EF%, Cavity Deformation %, Mean Wall Thickness, and the Final Abnormality Score.

This methodology strongly aligns the evaluation pipeline with the theoretical capabilities of cardiac MRI analysis, making the thesis's proposed framework robust and academically defensible.
