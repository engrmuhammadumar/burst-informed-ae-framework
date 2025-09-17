import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns, matplotlib.pyplot as plt
from .model_ensemble import stacked_meta_probs

def _evaluate_split(models, meta, weights, X, y):
    P = stacked_meta_probs(models, X, weights)
    y_hat = meta.predict(P)
    acc = accuracy_score(y, y_hat)
    f1 = f1_score(y, y_hat, average="macro")
    # ROC-AUC (ovr) from meta probs
    C = P.shape[1] // 3
    # average probs across base models
    proba = P.reshape(len(y), 3, C).sum(axis=1) / 3.0
    try:
        auc = roc_auc_score(pd.get_dummies(y).values, proba, multi_class="ovr")
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y, y_hat, labels=sorted(np.unique(y)))
    return {"acc": acc, "f1": f1, "auc": auc, "cm": cm, "y_hat": y_hat}

def evaluate_all(models, meta, weights, Xtr, ytr, Xv, yv, Xte, yte, outdir):
    res = {}
    for name, X, y in [("train", Xtr, ytr), ("val", Xv, yv), ("test", Xte, yte)]:
        r = _evaluate_split(models, meta, weights, X, y)
        res[name] = {k: (v.tolist() if hasattr(v, "tolist") else float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}
        # save confusion matrix image
        labels = sorted(np.unique(y))
        plt.figure(figsize=(4,3))
        sns.heatmap(r["cm"], annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout(); plt.savefig(outdir / f"confmat_{name}.png", dpi=150)
        plt.close()
    return res
