# Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines

This repository contains the official implementation of the paper:

> **Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines**  
> Engineering Failure Analysis, Elsevier, 2025.  

The framework provides a burst-informed, interpretable diagnostic pipeline for milling tools using acoustic emission (AE) signals. It combines burst-preserving signal processing, multi-domain feature extraction, and a performance-weighted stacking ensemble to deliver accurate and explainable fault diagnosis.

---

## 1. Project summary

Tool wear and fracture in milling operations generate high-frequency AE bursts that are easily masked by noise and stationary cutting activity. This framework:

- isolates fracture-related bursts while suppressing background noise,  
- extracts physically meaningful features across multiple domains,  
- learns a robust mapping from AE behaviour to fault classes, and  
- explains the final predictions using feature-level attribution.

The method has been validated on a precision milling test rig and an independent public AE dataset, showing high diagnostic accuracy and strong generalisation.

---

## 2. Method overview

The complete pipeline (BIWD-BAAFS-PWSE) consists of:

1. **Burst-Informed Wavelet Denoising (BIWD)**  
   - Hybrid soft–hard wavelet thresholding guided by local energy and noise statistics.  
   - Preserves short, high-energy bursts linked to crack initiation and tool damage.

2. **Burst-Aware Adaptive Frame Segmentation (BAAFS)**  
   - Energy-ratio-based burst detection and adaptive framing.  
   - Produces burst-centred segments while discarding non-informative regions.

3. **Multi-Domain Feature Extraction**  
   - Time-domain (TD): RMS, MAV, skewness, kurtosis, crest factor, etc.  
   - Frequency-domain (FD): dominant frequency, PSD, spectral entropy, THD, etc.  
   - Time–frequency (TFD): CWT-based descriptors, envelope area, wavelet energy.  
   - Higher-order statistics (HOS): higher-order moments and cumulants.  
   - Burst-specific: burst energy, peak amplitude, inter-burst interval, rise time, burst rate.

4. **Feature Engineering**  
   - Filter step using mutual information and correlation analysis.  
   - Embedded importance from ensemble learners with repeated cross-validation.  
   - Min–max scaling to obtain a compact, well-conditioned feature set.

5. **Performance-Weighted Stacked Ensemble (PWSE)**  
   - Base learners: **TabNet**, **XGBoost**, and **SVM**.  
   - Meta-learner: **Random Forest** operating on class-probability outputs.  
   - Base models are weighted according to validation performance.

6. **Explainability and Evaluation**  
   - Global and local explanations using **SHAP** and **LIME**.  
   - Standard metrics: Accuracy, Precision, Recall, F1-score, AUC, MCC, latency.  

---

## 3. Repository structure

The repository is organised as follows:

```text
burst-informed-ae-framework/
│
├─ Implemnetation/                 # Core implementation scripts (main pipeline)
├─ additional experiments/         # Cross-speed and robustness experiments
├─ comparisons/                    # Baseline models and comparative studies
├─ results/                        # Stored metrics, figures, and tables
│
├─ signal_processing.py            # BIWD denoising + burst-aware segmentation
├─ feature_engineering.py          # Multi-domain feature extraction and selection
├─ features.py                     # Utility functions for feature handling
├─ model_ensemble.py               # TabNet, XGBoost, SVM + RF meta-classifier
├─ interpretability.py             # SHAP and LIME analysis utilities
├─ evaluate.py                     # Metric calculations and plotting
├─ train.py                        # End-to-end training script
├─ main.py                         # Entry point for full pipeline execution
├─ config.yaml                     # Configuration (paths, parameters, model options)
│
└─ README.md                       # Project description (this file)
