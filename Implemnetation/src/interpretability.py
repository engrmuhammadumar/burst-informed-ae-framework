import numpy as np, pandas as pd
import shap, matplotlib.pyplot as plt

def shap_global(meta_model, P, out_path):
    explainer = shap.TreeExplainer(meta_model)
    sv = explainer.shap_values(P)
    shap.summary_plot(sv, P, show=False)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
