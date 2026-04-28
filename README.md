# Drift-Interaction Aware Transformer (DIAT)

## Overview
Drift-Interaction Aware Transformer (DIAT) is a deep learning framework for multivariate time-series forecasting under non-stationary conditions. The model is designed for Remaining Useful Life (RUL) prediction using the CMAPSS turbofan engine degradation dataset.

DIAT explicitly models the interaction between **distributional drift** and **periodic drift**, which are often ignored in traditional forecasting models.

---

## Problem Statement
Real-world time-series data is non-stationary, meaning both:
- Statistical distributions (mean, variance, correlations)
- Periodic/seasonal patterns

change over time. Most existing models treat these independently, leading to degraded performance under drift.

DIAT addresses this by jointly modeling both forms of drift within a transformer architecture.

---

## Key Features
- Models **distribution drift** using Kolmogorov–Smirnov (KS) statistics  
- Captures **periodic drift** using Fast Fourier Transform (FFT)  
- Drift-conditioned transformer encoder with adaptive attention gating  
- Fusion-based prediction head combining temporal + drift signals  
- PatchTST-style transformer backbone for multivariate forecasting  
- Designed for non-stationary industrial time-series data  

---

## Dataset
- **CMAPSS Turbofan Engine Degradation Dataset**
- Task: Remaining Useful Life (RUL) prediction
- Multivariate sensor signals from simulated aircraft engine degradation

---

## Methodology
1. **Input Representation**  
   Multivariate sensor time-series are segmented into sliding windows.

2. **Distribution Drift Estimation**  
   KS-test compares current window distribution with reference training distribution.

3. **Transformer Encoder**  
   Patch-based transformer processes input sequences with drift-conditioned attention.

4. **Periodic Drift Estimation**  
   FFT-based spectral analysis captures changes in periodic behavior.

5. **Fusion Head**  
   Combines encoded features with drift signals to predict RUL.

---

## Model Variants (Ablation Study)
- Baseline (no drift modeling)
- No Distribution Drift (no_D)
- No Periodic Drift (no_dP)
- Periodic-only model
- Full DIAT model
- Interaction-gated variant

---

## Results (CMAPSS FD001)
The proposed model improves performance under non-stationary conditions.

| Model | MAE | RMSE | NASA Score |
|------|-----|------|------------|
| Baseline | 10.91 | 15.13 | 492.8 |
| Full DIAT | 10.25 | 13.97 | 357.6 |

Key observation:
- Drift-aware modeling improves robustness under degradation
- Periodic-only modeling is insufficient for RUL prediction

---

## Tech Stack
- Python
- PyTorch
- NumPy / SciPy
- Scikit-learn
- Matplotlib

---

## Status
Ongoing research project (experiments and final analysis in progress)
