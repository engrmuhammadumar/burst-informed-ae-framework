import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

def filter_top_k(X, y, k=64):
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    idx = np.argsort(mi)[::-1][:k]
    return idx, mi

def embedded_select(X, y, model="xgb", final_dim=64, random_state=42):
    if model == "xgb":
        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            eval_metric="mlogloss", random_state=random_state
        )
        clf.fit(X, y)
        imp = clf.feature_importances_
    else:
        raise NotImplementedError
    idx = np.argsort(imp)[::-1][:final_dim]
    return idx, imp

def scale_fit_transform(X_train, X_val, X_test, method="minmax"):
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise NotImplementedError
    scaler.fit(X_train)
    return scaler, scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
