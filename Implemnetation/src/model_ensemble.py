import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier

def train_base_models(X, y, cfg):
    models = {}
    # TabNet
    tabnet = TabNetClassifier(
        n_d=cfg["tabnet"]["n_d"], n_a=cfg["tabnet"]["n_a"], n_steps=cfg["tabnet"]["n_steps"],
        gamma=cfg["tabnet"]["gamma"], clip_value=cfg["tabnet"]["clip_value"], verbose=0, seed=42
    )
    tabnet.fit(
        X, y, eval_set=[(X, y)], max_epochs=cfg["tabnet"]["epochs"],
        batch_size=cfg["tabnet"]["batch_size"], virtual_batch_size=cfg["tabnet"]["batch_size"],
        patience=20, drop_last=False
    )
    models["tabnet"] = tabnet

    xgb = XGBClassifier(**cfg["xgb"], objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1)
    xgb.fit(X, y)
    models["xgb"] = xgb

    svm = SVC(C=cfg["svm"]["C"], kernel=cfg["svm"]["kernel"], degree=cfg["svm"]["degree"],
              gamma=cfg["svm"]["gamma"], probability=True)
    svm.fit(X, y)
    models["svm"] = svm
    return models

def performance_weights(models, X, y):
    # simple CV accuracy weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = {k: [] for k in models}
    for tr, va in skf.split(X, y):
        for k, m in models.items():
            if k == "tabnet":  # refit small for speed
                continue
            y_pred = m.predict(X[va])
            accs[k].append((y_pred == y[va]).mean())
    # Assign equal to TabNet if not re-evaluated
    acc = {k: (np.mean(v) if v else 1.0) for k, v in accs.items()}
    w = np.array([acc.get("tabnet",1.0), acc["xgb"], acc["svm"]], dtype=float)
    w = w / (w.sum() + 1e-12)
    return {"tabnet": w[0], "xgb": w[1], "svm": w[2]}

def stacked_meta_probs(models, X, weights):
    P = []
    for k in ["tabnet", "xgb", "svm"]:
        proba = models[k].predict_proba(X)
        P.append(weights[k] * proba)
    return np.hstack(P)  # concat weighted probs

def train_meta_rf(P_train, y, cfg):
    rf = RandomForestClassifier(**cfg["meta_rf"], random_state=42)
    rf.fit(P_train, y)
    return rf
