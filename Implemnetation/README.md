# Burst-Informed AE – Explainable Failure Diagnosis (BF/GF/N/TF)

Paper-aligned pipeline:
1) **Wavelet denoising (hybrid soft–hard)** and **burst-aware segmentation**  
2) **Multi-domain features**: TD, FD, TFD (CWT), HOS, Burst  
3) **Hybrid feature selection** (mutual info → XGBoost importances) + MinMax scaling  
4) **Performance-weighted ensemble**: TabNet, XGBoost, SVM → RF meta-classifier  
5) **Metrics & Explainability**: Accuracy, F1, AUC, confusion matrix, SHAP




