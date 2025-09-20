# Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines

This repository provides the official implementation of our paper:

**“Burst-Informed Acoustic Emission Framework for Explainable Failure Diagnosis in Milling Machines”**  

---

## 🔹 Overview

Tool wear and failure in milling operations can be challenging to diagnose due to noise, burst transients, and the multi-domain nature of acoustic emission (AE) signals.  
This framework introduces a **burst-aware, explainable AE-based diagnostic pipeline** that integrates:

1. **Hybrid Wavelet Denoising** – adaptive soft–hard shrinkage to remove high-frequency noise  
2. **Burst-Informed Segmentation** – energy-based detection of bursts and stationary cutting intervals  
3. **Multi-Domain Feature Extraction** – time, frequency, time–frequency (CWT), higher-order statistics, and burst-specific features  
4. **Hybrid Feature Selection** – mutual information + XGBoost importance with MinMax scaling  
5. **Performance-Weighted Ensemble** – base learners (TabNet, XGBoost, SVM) combined via Random Forest meta-classifier  
6. **Explainability Module** – SHAP-based interpretation of features and ensemble contributions  

The framework provides **robust, reproducible, and interpretable failure diagnosis** across tool conditions.

