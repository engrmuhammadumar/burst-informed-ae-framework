# Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines

This repository provides the official implementation of our paper:

**“Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines”**  

---

## Overview

Tool wear and failure in milling operations can be challenging to diagnose due to noise, burst transients, and the multi-domain nature of acoustic emission (AE) signals.  
This framework introduces a **burst-aware, explainable AE-based diagnostic pipeline** that integrates:

1. **Hybrid Wavelet Denoising** – adaptive soft–hard shrinkage to remove high-frequency noise  
2. **Burst-Informed Segmentation** – energy-based detection of bursts and stationary cutting intervals  
3. **Multi-Domain Feature Extraction** – time, frequency, time–frequency (CWT), higher-order statistics, and burst-specific features  
4. **Hybrid Feature Selection** 
5. **Performance-Weighted Ensemble** 
6. **Explainability Module**

The framework provides **robust, reproducible, and interpretable failure diagnosis** across tool conditions.


**Files**


signal_processing.py → Burst-informed preprocessing (wavelet denoising + burst segmentation).

feature_engineering.py → Multi-domain feature extraction (time, frequency, time–frequency, higher-order stats, bursts).

features.py → Utility functions for feature handling.

model_ensemble.py → Ensemble learning (TabNet, XGBoost, SVM + RF meta-classifier).

interpretability.py → Explainability using SHAP and LIME.

evaluate.py → Performance metrics (Accuracy, F1, AUC, Confusion Matrix).

train.py → End-to-end model training script.

main.py → Entry point that ties everything together.

config.yaml → Configuration file (paths, parameters, model settings).
