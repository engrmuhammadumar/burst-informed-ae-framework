import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .io import load_index, load_signal
from .signal_processing import wavelet_denoise, burst_segments
from .features import extract_all_domains
from .feature_engineering import filter_top_k, embedded_select, scale_fit_transform
from .model_ensemble import train_base_models, performance_weights, stacked_meta_probs, train_meta_rf
from .evaluate import evaluate_all
import joblib, json, random

def _seed_everything(seed=42):
    import os, torch
    random.seed(seed); np.random.seed(seed)
    try:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)

def build_feature_table(cfg):
    df = load_index(cfg["data"]["index_csv"], cfg["data"]["label_column"])
    fs = cfg["sampling_rate_hz"]
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        sig = load_signal(r[cfg["data"]["signal_column"]], cfg["data"]["file_type"]).astype(float)
        sig = wavelet_denoise(sig, wavelet=cfg["preprocess"]["wavelet"],
                              level=cfg["preprocess"]["level"],
                              alpha=cfg["preprocess"]["soft_hard_alpha"])
        segs = burst_segments(sig, fs,
                              win_ms=cfg["preprocess"]["burst"]["win_ms"],
                              alpha=cfg["preprocess"]["burst"]["alpha"],
                              pad_ms=cfg["preprocess"]["burst"]["pad_ms"],
                              max_segments=cfg["preprocess"]["burst"]["max_segments"])
        feats = extract_all_domains(
            sig, fs, segs, cfg["features"]["tfd"]
        )
        feats["label"] = r[cfg["data"]["label_column"]]
        rows.append(feats)
    feat_df = pd.DataFrame(rows)
    return feat_df

def run_full_pipeline(cfg):
    _seed_everything(cfg["seed"])
    outdir = Path(cfg["logging"]["out_dir"]); outdir.mkdir(parents=True, exist_ok=True)

    feat_path = outdir / "features.parquet"
    if feat_path.exists():
        feat_df = pd.read_parquet(feat_path)
    else:
        feat_df = build_feature_table(cfg)
        feat_df.to_parquet(feat_path)

    X = feat_df.drop(columns=["label"]).values.astype(np.float32)
    y = feat_df["label"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg["train"]["test_size"], stratify=y if cfg["train"]["stratify"] else None, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg["train"]["val_size"], stratify=y_trainval if cfg["train"]["stratify"] else None, random_state=42
    )

    # Hybrid selection
    idx_filt, mi = filter_top_k(X_train, y_train, k=cfg["selection"]["top_k_filter"])
    X_train_f = X_train[:, idx_filt]; X_val_f = X_val[:, idx_filt]; X_test_f = X_test[:, idx_filt]

    idx_emb, imp = embedded_select(X_train_f, y_train, model=cfg["selection"]["embedded_model"],
                                   final_dim=cfg["selection"]["final_dim"])
    idx_final = idx_filt[idx_emb]
    X_train_s = X_train[:, idx_final]; X_val_s = X_val[:, idx_final]; X_test_s = X_test[:, idx_final]

    # Scale
    scaler, X_train_s, X_val_s, X_test_s = scale_fit_transform(X_train_s, X_val_s, X_test_s, method=cfg["train"]["scaler"])
    joblib.dump({"idx": idx_final, "scaler": scaler}, outdir / "preproc.joblib")

    # Train ensemble
    models = train_base_models(X_train_s, y_train, cfg["ensemble"])
    weights = performance_weights(models, X_val_s, y_val)
    P_tr = stacked_meta_probs(models, X_train_s, weights)
    meta = train_meta_rf(P_tr, y_train, cfg["ensemble"])

    # Save
    joblib.dump({"models": models, "weights": weights, "meta": meta}, outdir / "ensemble.joblib")

    # Evaluate
    res = evaluate_all(models, meta, weights, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, outdir)
    return res
