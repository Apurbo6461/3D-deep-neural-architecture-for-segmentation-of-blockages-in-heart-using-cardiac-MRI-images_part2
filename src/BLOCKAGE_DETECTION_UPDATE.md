# Multi-Evidence Cardiac Dysfunction Detection — Methodology Update

The detection methodology has been completely overhauled from a simple single-threshold approach to a robust, multi-evidence hybrid framework that dramatically reduces false positives in healthy (NOR) patients.

## 1. Pipeline Overhaul (ED vs ES)

For each patient, the pipeline explicitly extracts the **ED frame** and **ES frame** (identified via `Info.cfg`). Both frames are passed through the deep learning models to predict segmentations, which are then compared to derive functional metrics.

## 2. AHA 6-Region Averaging (Noise Reduction)

**Problem:** Pointwise distance transforms on imperfect segmentations produce noisy local thickness measurements, leading to false positives.

**Solution:** The myocardium is divided into **6 circumferential sectors** (simplified AHA model) by computing the angular position of each voxel relative to the myocardium centroid. Wall thickening is computed as the **mean per sector**, not raw voxel-level values.

- A sector is "impaired" only if its **mean regional thickening** falls below the 35% threshold
- This eliminates spurious voxel-level noise and aligns with standard cardiac function analysis practices

## 3. Multi-Evidence Hybrid Framework

Instead of flagging dysfunction based on a single metric, the new framework requires **converging evidence** from multiple independent checks:

### Evidence Items

| # | Evidence | Condition | Score |
|---|----------|-----------|-------|
| A | Regional wall thickening | ≥1 AHA region impaired (<35%) | +1 |
| B | Ejection Fraction / Cavity Deformation | EF < 45% | +1 |
| C | ED wall thickness | Structurally thin (< ~5mm) | +1 |
| D | Multi-region convergence | ≥3 AHA regions impaired | +1 |

### Risk Stratification

| Score | Classification | Meaning |
|-------|---------------|---------|
| 0–1 | **Normal** | Isolated noise or minor variation — no clinical concern |
| 2 | **Suspicious** | Two independent lines of evidence — warrants monitoring |
| 3–4 | **High Suspicion** | Strong multi-metric evidence of myocardial dysfunction |

### Why This Eliminates NOR False Positives

Normal (NOR) patients may occasionally show borderline thickening in one region due to segmentation noise, giving them a score of 1. However, they will have:
- Normal EF (>50%) → no score from Evidence B
- Normal wall thickness → no score from Evidence C
- At most 1 region impaired → no score from Evidence D

Therefore NOR patients almost never reach a suspicion score ≥ 2.

## 4. EF and Cavity Deformation: Clarification

Both metrics are derived from the same LV volumes:

```
EF = (EDV - ESV) / EDV × 100
Cavity Deformation = (EDV_voxels - ESV_voxels) / EDV_voxels × 100
```

They are **mathematically equivalent** and are therefore counted as a **single evidence item** (not two separate scores). Both are reported for clinical readability.

## 5. Terminology

Throughout the codebase:
- "Blockage" → "Myocardial Dysfunction" / "Contractile Impairment"
- "Blockage Rate" → "Dysfunction Rate"
- "Blockage Map" → "Contractile Impairment Map"

This avoids the misleading implication that we are directly detecting coronary artery blockages (which MRI cannot do). Instead, we detect **functional abnormalities** (reduced contraction, wall thinning) that may be **associated with** blockage-induced ischemia, but can also arise from DCM, HCM, or other cardiomyopathies.

## 6. Enhanced Visualizations

- **AHA 6-region bar chart** added to patient visualizations showing per-sector thickening with threshold line
- **Risk level** prominently displayed instead of raw scores
- **Per-region impairment labels** on anatomical subplots

## 7. NOR vs MINF Validation

A dedicated validation script (`nor_vs_minf_validation.py`) compares all 10 NOR patients against all 10 MINF patients, producing:
- Box plots of wall thickening, EF, and suspicion score distributions
- Bar charts of impaired region counts and score distributions
- Scatter plot of thickening vs EF per patient

This provides strong quantitative evidence that the framework correctly distinguishes healthy hearts from infarcted hearts.
