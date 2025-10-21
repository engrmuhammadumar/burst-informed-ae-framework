# ae_pipeline_660_plus.py
# Author: UMAR + ChatGPT
# Description: End-to-end AE feature pipeline + CV-weighted TabNet/SVM/RF/(XGB) ensemble
# Inputs: raw AE folders or a precomputed full_ae_features.xlsx
# Outputs: features Excel, confusion matrix (PNG), ROC (PNG), t-SNE (PNG), reports (txt/csv), tsne_coordinates.csv, feature_importance.csv

import os
import sys
import json
import time
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

# ========== Utilities: ensure packages ==========
def ensure(package, import_name=None, extras=None):
    try:
        __import__(import_name or package)
    except ImportError:
        pkg_str = package if extras is None else f"{package}[{extras}]"
        print(f"[setup] Installing missing package: {pkg_str} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_str])
        __import__(import_name or package)

# Core stack
ensure("numpy")
ensure("scipy", "scipy")
ensure("pandas")
ensure("matplotlib", "matplotlib")
ensure("seaborn")
ensure("scikit-learn", "sklearn")
ensure("PyWavelets", "pywt")
ensure("pytorch_tabnet", "pytorch_tabnet.tab_model")
ensure("h5py")
# Optional, boosts accuracy often; safe to skip if install fails
try:
    ensure("xgboost")
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import pywt
import h5py
from scipy.io import loadmat
from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis, entropy

from pytorch_tabnet.tab_model import TabNetClassifier

# ========== CONFIG ==========
# Your 2024-09-25 / 660 data paths
CLASS_DIRS = {
    "BF": r"F:\20240925\BF660_1\AE",
    "GF": r"F:\20240925\GF660_1\AE",
    "TF": r"F:\20240925\TF660_1\AE",
    "N" : r"F:\20240925\N660_1\AE",
}
INCLUDE_SECOND_BATCH = True
SECOND_BATCH_MAP = {
    "BF": r"F:\20240925\BF660_2\AE",
    "GF": r"F:\20240925\GF660_2\AE",
    "TF": r"F:\20240925\TF660_2\AE",
    "N" : r"F:\20240925\N660_2\AE",
}

# If you already have features, set path here to SKIP raw extraction:
FEATURES_FILE = None  # e.g., r"E:\...\full_ae_features.xlsx"  # <- set if you want to load existing features

# Sampling frequency
FS = 1_000_000

# Segmentation
FRAME_SIZE = 10_000
NUM_FRAMES_PER_SIGNAL = 10

# Wavelet denoise
WAVELET = "db4"
W_LEVEL = 3

# Burst selection
BURST_TOP_RATIO = 0.5

# CWT TFD
TFD_WAVELET = "cmor1.5-1.0"
TFD_SCALES = 64

# Split & randomness
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Output folder
OUTDIR = Path("./ae_outputs_20240925_660_plus")
OUTDIR.mkdir(parents=True, exist_ok=True)

# CV / tuning knobs (aim higher accuracy)
USE_CV_TUNING = True
CV_FOLDS = 5

# ========== IO Helpers ==========
def list_files_recursive(root_dir, patterns=(".mat", ".npy", ".csv", ".txt")):
    files = []
    root_dir = Path(root_dir)
    if not root_dir.exists():
        print(f"[warn] Folder not found: {root_dir}")
        return files
    for ext in patterns:
        files += [str(p) for p in root_dir.rglob(f"*{ext}")]
    return sorted(files)

def load_signal_from_file(path):
    p = Path(path)
    ext = p.suffix.lower()
    sigs = []
    try:
        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            arr = np.array(arr)
            if arr.ndim == 1:
                sigs = [arr.astype(float)]
            elif arr.ndim == 2:
                if arr.shape[0] >= arr.shape[1]:
                    for i in range(arr.shape[1]):
                        sigs.append(arr[:, i].astype(float))
                else:
                    for i in range(arr.shape[0]):
                        sigs.append(arr[i, :].astype(float))
            else:
                sigs = [arr.flatten().astype(float)]

        elif ext in (".csv", ".txt"):
            arr = np.loadtxt(path, dtype=float, delimiter="," if ext == ".csv" else None)
            arr = np.array(arr, dtype=float)
            if arr.ndim == 1:
                sigs = [arr]
            elif arr.ndim == 2:
                if arr.shape[0] >= arr.shape[1]:
                    for i in range(arr.shape[1]):
                        sigs.append(arr[:, i])
                else:
                    for i in range(arr.shape[0]):
                        sigs.append(arr[i, :])
            else:
                sigs = [arr.flatten()]

        elif ext == ".mat":
            try:
                md = loadmat(path)
                candidates = []
                for k, v in md.items():
                    if k.startswith("__"):
                        continue
                    arr = np.array(v)
                    if arr.ndim in (1, 2):
                        candidates.append((k, arr.size, arr.shape))
                if not candidates:
                    raise ValueError("No 1D/2D variables in MAT (v7).")
                candidates.sort(key=lambda x: x[1], reverse=True)
                key = candidates[0][0]
                arr = np.array(md[key]).squeeze()
                if arr.ndim == 1:
                    sigs = [arr.astype(float)]
                elif arr.ndim == 2:
                    if arr.shape[0] >= arr.shape[1]:
                        for i in range(arr.shape[1]):
                            sigs.append(arr[:, i].astype(float))
                    else:
                        for i in range(arr.shape[0]):
                            sigs.append(arr[i, :].astype(float))
            except Exception:
                with h5py.File(path, "r") as f:
                    def walk_dsets(h5obj, prefix=""):
                        for k in h5obj.keys():
                            obj = h5obj[k]
                            name = f"{prefix}/{k}".strip("/")
                            if isinstance(obj, h5py.Dataset):
                                yield name, obj
                            elif isinstance(obj, h5py.Group):
                                yield from walk_dsets(obj, name)
                    best = None
                    for name, ds in walk_dsets(f):
                        if best is None or ds.size > best[1].size:
                            best = (name, ds)
                    if best is None:
                        raise ValueError("No datasets found in MAT v7.3 file")
                    arr = np.array(best[1])
                    if arr.ndim == 1:
                        sigs = [arr.astype(float)]
                    elif arr.ndim == 2:
                        if arr.shape[0] >= arr.shape[1]:
                            for i in range(arr.shape[1]):
                                sigs.append(arr[:, i].astype(float))
                        else:
                            for i in range(arr.shape[0]):
                                sigs.append(arr[i, :].astype(float))
                    else:
                        sigs = [arr.flatten().astype(float)]
        else:
            print(f"[warn] Unsupported extension: {ext} ({path})")
    except Exception as e:
        print(f"[error] Failed to load {path}: {e}")

    cleaned = []
    for s in sigs:
        s = np.array(s, dtype=float).flatten()
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        cleaned.append(s)
    return cleaned

# ========== Preprocessing ==========
def segment_signal(signal, frame_size=FRAME_SIZE, num_frames=NUM_FRAMES_PER_SIGNAL):
    segments = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if end <= len(signal):
            segments.append(signal[start:end])
    return segments

def denoise_wavelet(frame, wavelet=WAVELET, level=W_LEVEL):
    coeffs = pywt.wavedec(frame, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(frame)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    den = pywt.waverec(coeffs_thresh, wavelet=wavelet)
    return den[:len(frame)]

def compute_energy(frame):
    return np.sum(frame ** 2)

def select_burst_frames(frames, top_ratio=BURST_TOP_RATIO):
    energies = np.array([compute_energy(f) for f in frames])
    if len(energies) == 0:
        return [], np.array([]), 0.0
    thr = np.percentile(energies, 100 * (1 - top_ratio))
    selected = [f for f, e in zip(frames, energies) if e >= thr]
    return selected, energies, thr

# ========== Feature Extraction ==========
def extract_td_fd_features(frame, fs=FS):
    features = []
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    var_val = np.var(frame)
    rms_val = np.sqrt(np.mean(frame**2))
    peak_val = np.max(np.abs(frame))
    ptp_val = np.ptp(frame)
    crest_factor = peak_val / rms_val if rms_val != 0 else 0
    mav = np.mean(np.abs(frame))
    impulse_factor = peak_val / mav if mav != 0 else 0
    shape_factor = rms_val / mav if mav != 0 else 0
    margin_factor = peak_val / (np.mean(np.sqrt(np.abs(frame)))**2) if np.mean(np.sqrt(np.abs(frame))) != 0 else 0
    skewness = skew(frame)
    kurtv = kurtosis(frame)
    energy = np.sum(frame**2)
    zcr = ((frame[:-1] * frame[1:]) < 0).sum()
    max_val = np.max(frame)
    min_val = np.min(frame)
    range_val = max_val - min_val
    duration = len(frame) / fs
    envelope_area = np.trapz(np.abs(frame))

    freqs, psd = welch(frame, fs=fs, nperseg=1024)
    psd = np.nan_to_num(psd)
    spectral_energy = np.sum(psd)
    spectral_centroid = (np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) != 0 else 0
    spectral_entropy = entropy(psd)
    spectral_kurtosis = kurtosis(psd)
    spectral_skewness = skew(psd)
    spectral_flatness = (np.exp(np.mean(np.log(psd + 1e-12))) / np.mean(psd)) if np.mean(psd) != 0 else 0
    dominant_freq = freqs[np.argmax(psd)]
    cumsum_psd = np.cumsum(psd)
    median_freq = freqs[np.where(cumsum_psd >= np.sum(psd) / 2)[0][0]] if np.sum(psd) > 0 else 0.0
    mean_freq = spectral_centroid
    freq_variance = np.var(psd)
    freq_rms = np.sqrt(np.mean(psd**2))
    freq_spread = np.std(freqs)
    harmonics, _ = find_peaks(psd, height=np.max(psd)*0.1 if psd.size else None)
    thd = np.sum(psd[harmonics[1:]]) / psd[harmonics[0]] if len(harmonics) > 1 else 0

    features.extend([
        mean_val, rms_val, std_val, var_val, skewness, kurtv, peak_val, ptp_val,
        crest_factor, impulse_factor, shape_factor, margin_factor,
        energy, max_val, min_val, range_val, zcr, duration, mav, envelope_area,
        mean_freq, median_freq, dominant_freq, spectral_centroid, spectral_entropy,
        spectral_kurtosis, spectral_skewness, spectral_flatness, spectral_energy,
        freq_variance, freq_rms, freq_spread, thd
    ])
    return features

def extract_tfd_features_cwt(frame, wavelet=TFD_WAVELET, fs=FS, num_scales=TFD_SCALES):
    scales = np.arange(1, num_scales + 1)
    coef, _ = pywt.cwt(frame, scales=scales, wavelet=wavelet, sampling_period=1/fs)
    scalogram = np.abs(coef)
    tfd_energy = np.sum(scalogram**2)
    tfd_entropy = entropy(scalogram.flatten())
    tfd_mean = np.mean(scalogram)
    tfd_std = np.std(scalogram)
    tfd_max = np.max(scalogram)
    tfd_kurt = kurtosis(scalogram.flatten())
    tfd_skew = skew(scalogram.flatten())
    tfd_centroid = np.sum(scalogram * np.arange(scalogram.shape[0])[:, None]) / (np.sum(scalogram) + 1e-12)
    flat = scalogram.flatten()
    top_k = np.percentile(flat, 95)
    ridge_energy = np.sum(flat[flat >= top_k]) / (np.sum(flat) + 1e-12)
    return [tfd_energy, tfd_entropy, tfd_mean, tfd_std, tfd_max, tfd_kurt, tfd_skew, tfd_centroid, ridge_energy]

def extract_hos_features(frame):
    centered = frame - np.mean(frame)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    m5 = np.mean(centered**5)
    c3 = m3
    c4 = m4 - 3 * (np.var(frame)**2)
    g_index = c4 / m4 if m4 != 0 else 0
    analytic = hilbert(frame)
    N = len(frame) // 2
    if N < 4:
        return [m3, m4, m5, c3, c4, g_index, 0, 0, c3**2 + c4**2]
    outer = np.outer(analytic[:N], analytic[:N])
    X = np.fft.fft2(outer)
    bispec = np.abs(X)
    denom = (np.outer(np.abs(analytic[:N])**2, np.abs(analytic[:N])**2) + 1e-12)
    bicoherence = bispec / denom
    bispec_mean = np.mean(bispec)
    bicoherence_mean = np.mean(bicoherence)
    nonlinearity_index = c3**2 + c4**2
    return [m3, m4, m5, c3, c4, g_index, bispec_mean, bicoherence_mean, nonlinearity_index]

def extract_burst_features(frame, fs=FS, threshold_ratio=0.25):
    threshold = threshold_ratio * np.max(np.abs(frame)) if len(frame) else 0.0
    idx = np.where(np.abs(frame) > threshold)[0] if threshold > 0 else np.array([])
    if len(idx) == 0:
        return [0]*10
    starts = [idx[0]]
    durations = []
    cur = 1
    for i in range(1, len(idx)):
        if idx[i] == idx[i-1] + 1:
            cur += 1
        else:
            durations.append(cur / fs)
            starts.append(idx[i])
            cur = 1
    durations.append(cur / fs)
    energies = [np.sum(frame[s:s+int(d*fs)]**2) for s, d in zip(starts, durations)]
    peaks = [np.max(np.abs(frame[s:s+int(d*fs)])) for s, d in zip(starts, durations)]
    rmss = [np.sqrt(np.mean(frame[s:s+int(d*fs)]**2)) for s, d in zip(starts, durations)]
    ibis = np.diff(starts) / fs if len(starts) > 1 else np.array([])
    total_event_duration = np.sum(durations)
    cumulative_counts = np.sum([int(d*fs) for d in durations])
    return [
        len(durations),
        float(np.mean(durations)) if len(durations) else 0.0,
        float(np.mean(ibis)) if len(ibis) else 0.0,
        float(np.mean(energies)) if len(energies) else 0.0,
        float(np.mean(peaks)) if len(peaks) else 0.0,
        float(np.mean(rmss)) if len(rmss) else 0.0,
        int(cumulative_counts),
        float(total_event_duration),
        float(np.max(peaks)) if len(peaks) else 0.0,
        float(np.min(peaks)) if len(peaks) else 0.0
    ]

# ========== t-SNE (version-proof) ==========
def make_tsne(perplexity=30, learning_rate=200, random_state=RANDOM_STATE, verbose=1):
    try:
        return TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate,
            n_iter=1000, verbose=verbose, random_state=random_state,
            method="barnes_hut", angle=0.5
        )
    except TypeError:
        return TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate,
            verbose=verbose, random_state=random_state,
            method="barnes_hut", angle=0.5
        )

# ========== Pipeline ==========
def main():
    print("=== AE Pipeline (660+) Start ===")
    start_time = time.time()

    # Option A: load features directly
    if FEATURES_FILE and Path(FEATURES_FILE).exists():
        print(f"[info] Loading features from: {FEATURES_FILE}")
        features_df = pd.read_excel(FEATURES_FILE)
        assert "Label" in features_df.columns, "Expected a 'Label' column in features file."
    else:
        # Option B: build features from raw AE folders
        effective_dirs = {k: [v] for k, v in CLASS_DIRS.items()}
        if INCLUDE_SECOND_BATCH:
            for k, v2 in SECOND_BATCH_MAP.items():
                if k in effective_dirs:
                    effective_dirs[k].append(v2)
                else:
                    effective_dirs[k] = [v2]

        raw_signals = {k: [] for k in effective_dirs.keys()}
        for cls, dir_list in effective_dirs.items():
            if not isinstance(dir_list, (list, tuple)):
                dir_list = [dir_list]
            for d in dir_list:
                files = list_files_recursive(d)
                print(f"[{cls}] Found {len(files)} files in {d}")
                for fp in files:
                    sigs = load_signal_from_file(fp)
                    raw_signals[cls].extend(sigs)
            print(f"[{cls}] Total signals loaded: {len(raw_signals[cls])}")

        burst_selected_data, burst_stats, per_class_counts = {}, {}, {}
        for cls, signals in raw_signals.items():
            frames_all = []
            for s in signals:
                segments = segment_signal(s, frame_size=FRAME_SIZE, num_frames=NUM_FRAMES_PER_SIGNAL)
                segments = [seg for seg in segments if len(seg) == FRAME_SIZE]
                segments = [denoise_wavelet(seg) for seg in segments]
                frames_all.extend(segments)
            selected, energies, thr = select_burst_frames(frames_all, top_ratio=BURST_TOP_RATIO)
            burst_selected_data[cls] = selected
            burst_stats[cls] = {"total_frames": len(frames_all), "selected_frames": len(selected), "energy_threshold": float(thr)}
            per_class_counts[cls] = len(selected)
            print(f"[{cls}] frames: {len(frames_all)} → selected: {len(selected)} (thr={thr:.2e})")

        td_fd_features, tfd_features, hos_features, burst_features, labels = [], [], [], [], []
        for cls, frames in burst_selected_data.items():
            for frame in frames:
                td_fd_features.append(extract_td_fd_features(frame))
                tfd_features.append(extract_tfd_features_cwt(frame))
                hos_features.append(extract_hos_features(frame))
                burst_features.append(extract_burst_features(frame))
                labels.append(cls)

        td_fd_names = [
            "Mean","RMS","STD","Variance","Skewness","Kurtosis","Peak","PeakToPeak","CrestFactor",
            "ImpulseFactor","ShapeFactor","MarginFactor","SignalEnergy","MaxVal","MinVal","Range",
            "ZCR","Duration","MAV","EnvelopeArea","MeanFreq","MedianFreq","DominantFreq",
            "SpectralCentroid","SpectralEntropy","SpectralKurtosis","SpectralSkewness",
            "SpectralFlatness","SpectralEnergy","FreqVariance","FreqRMS","FreqSpread","THD"
        ]
        tfd_names = ["TFD_Energy","TFD_Entropy","TFD_Mean","TFD_STD","TFD_Max","TFD_Kurtosis","TFD_Skewness","TFD_Centroid","TFD_RidgeEnergyRatio"]
        hos_names = ["Moment3","Moment4","Moment5","Cumulant3","Cumulant4","GaussianityIndex","BispectrumMean","BicoherenceMean","NonlinearityIndex"]
        burst_names = ["NumBursts","AvgBurstDuration","AvgInterBurstInterval","AvgBurstEnergy","AvgBurstPeak","AvgBurstRMS","CumulativeCounts","TotalEventDuration","MaxBurstPeak","MinBurstPeak"]

        td_fd_df = pd.DataFrame(td_fd_features, columns=td_fd_names)
        tfd_df   = pd.DataFrame(tfd_features, columns=tfd_names)
        hos_df   = pd.DataFrame(hos_features, columns=hos_names)
        burst_df = pd.DataFrame(burst_features, columns=burst_names)

        features_df = pd.concat([td_fd_df, tfd_df, hos_df, burst_df], axis=1)
        features_df["Label"] = labels

        features_path_raw = OUTDIR / "full_ae_features.xlsx"
        features_df.to_excel(features_path_raw, index=False)

    # ===== Modeling =====
    if len(features_df) == 0:
        raise RuntimeError("No features extracted/loaded.")

    # Separate X/y
    X = features_df.drop(columns=["Label"]).values
    y = features_df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
    )

    # Class weights (helps class imbalance)
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): w for c, w in zip(classes, cw)}

    # ---- TabNet (fast settings; can be tuned further)
    X_train_tab = X_train.astype(np.float32)
    X_test_tab  = X_test.astype(np.float32)
    tabnet = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
    tabnet.fit(X_train_tab, y_train, max_epochs=100, patience=10)

    # ---- SVM (RBF) with small CV grid (fast)
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    if USE_CV_TUNING:
        param_grid = {
            "C": [3, 10, 30, 100],
            "gamma": ["scale", 1e-3, 3e-4, 1e-4]
        }
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        svm = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, scoring="f1_weighted", refit=True, verbose=0)
    svm.fit(X_train, y_train)

    # ---- Random Forest with small CV grid
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_estimators=400)
    if USE_CV_TUNING:
        rf_grid = {
            "n_estimators": [400, 800],
            "max_depth": [None, 15, 25],
            "min_samples_split": [2, 5]
        }
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        rf = GridSearchCV(rf, rf_grid, cv=cv, n_jobs=-1, scoring="f1_weighted", refit=True, verbose=0)
    rf.fit(X_train, y_train)

    # ---- Optional XGBoost
    xgb_clf = None
    if HAS_XGB:
        from xgboost import XGBClassifier
        xgb_clf = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, objective="multi:softprob", num_class=len(class_names),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="mlogloss"
        )
        if USE_CV_TUNING:
            # keep it light; the defaults above are already strong
            pass
        xgb_clf.fit(X_train, y_train)

    # ===== Build CV weights by macro AUC on train (out-of-fold) =====
    def oof_probs(model, X_tr, y_tr, as_tabnet=False):
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros((len(y_tr), len(class_names)))
        for tr_idx, va_idx in cv.split(X_tr, y_tr):
            X_tr_i, X_va_i = X_tr[tr_idx], X_tr[va_idx]
            y_tr_i = y_tr[tr_idx]
            if as_tabnet:
                mdl = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
                mdl.fit(X_tr_i.astype(np.float32), y_tr_i, max_epochs=60, patience=8)
                oof[va_idx] = mdl.predict_proba(X_va_i.astype(np.float32))
            else:
                mdl = model
                # fresh clone where applicable
                try:
                    from sklearn.base import clone
                    mdl = clone(model)
                except Exception:
                    pass
                mdl.fit(X_tr_i, y_tr_i)
                oof[va_idx] = mdl.predict_proba(X_va_i)
        return oof

    def macro_auc(y_true_int, probs):
        yb = label_binarize(y_true_int, classes=list(range(len(class_names))))
        fpr, tpr = {}, {}
        aucs = []
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(yb[:, i], probs[:, i])
            aucs.append(auc(fpr[i], tpr[i]))
        return float(np.mean(aucs))

    print("\n[weights] Estimating model weights from CV macro AUC ...")
    tab_oof = oof_probs(tabnet, X_train, y_train, as_tabnet=True)
    tab_auc = macro_auc(y_train, tab_oof)

    # get best estimators for tuned models
    svm_fit = svm.best_estimator_ if isinstance(svm, GridSearchCV) else svm
    rf_fit  = rf.best_estimator_ if isinstance(rf, GridSearchCV) else rf

    svm_oof = oof_probs(svm_fit, X_train, y_train)
    rf_oof  = oof_probs(rf_fit,  X_train, y_train)
    sv_auc  = macro_auc(y_train, svm_oof)
    rf_auc  = macro_auc(y_train, rf_oof)

    if xgb_clf is not None:
        xgb_oof = oof_probs(xgb_clf, X_train, y_train)
        xg_auc  = macro_auc(y_train, xgb_oof)
    else:
        xg_auc = 0.0

    aucs = np.array([tab_auc, sv_auc, rf_auc, xg_auc], dtype=float)
    aucs = np.clip(aucs, 1e-6, None)
    weights = aucs / aucs.sum()
    if xgb_clf is None:
        weights = weights[:3] / weights[:3].sum()  # renormalize if no xgb
    print(f"[weights] TabNet={tab_auc:.4f}, SVM={sv_auc:.4f}, RF={rf_auc:.4f}" + (f", XGB={xg_auc:.4f}" if xgb_clf else ""))
    print(f"[weights] Using weights: {weights}")

    # ===== Test-time prediction (weighted soft voting) =====
    tabnet_probs = tabnet.predict_proba(X_test_tab)
    svm_probs    = svm_fit.predict_proba(X_test)
    rf_probs     = rf_fit.predict_proba(X_test)
    if xgb_clf is not None:
        xgb_probs = xgb_clf.predict_proba(X_test)
        avg_probs = weights[0]*tabnet_probs + weights[1]*svm_probs + weights[2]*rf_probs + weights[3]*xgb_probs
    else:
        avg_probs = weights[0]*tabnet_probs + weights[1]*svm_probs + weights[2]*rf_probs

    y_pred = np.argmax(avg_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy (Weighted Ensemble): {acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\n[RESULT] Classification Report:\n", report)

    cm = confusion_matrix(y_test, y_pred)

    # ===== Save reports =====
    features_path = OUTDIR / "full_ae_features_used.xlsx"
    features_used = pd.DataFrame(X_scaled, columns=[c for c in features_df.columns if c != "Label"])
    features_used["Label"] = [class_names[i] for i in le.transform(features_df["Label"].values)]
    features_used.to_excel(features_path, index=False)

    (OUTDIR / "metadata.json").write_text(json.dumps({
        "class_names": class_names,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "fs": FS,
        "frame_size": FRAME_SIZE,
        "num_frames_per_signal": NUM_FRAMES_PER_SIGNAL,
        "include_second_batch": INCLUDE_SECOND_BATCH,
        "cv_folds": CV_FOLDS,
        "use_cv_tuning": USE_CV_TUNING,
        "weights": weights.tolist()
    }, indent=2))

    (OUTDIR / "classification_report.txt").write_text(f"Accuracy: {acc:.6f}\n\n{report}")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUTDIR / "confusion_matrix.csv")

    # Confusion matrix PNG
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 18, "fontweight": "bold"})
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.tight_layout()
    cm_path = OUTDIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ROC curves
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], avg_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), avg_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7.5, 6.5))
    plt.plot(fpr["micro"], tpr["micro"], linewidth=2, label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linewidth=2, label=f"macro-average (AUC = {roc_auc['macro']:.3f})")
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], linewidth=1.5, label=f"{name} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=14, fontweight="bold")
    plt.title("Ensemble ROC (One-vs-Rest)", fontsize=16, fontweight="bold")
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    roc_path = OUTDIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    with open(OUTDIR / "roc_auc_report.txt", "w") as f:
        f.write("ROC AUC per class (ensemble weighted):\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: {roc_auc[i]:.6f}\n")
        f.write(f"\nMicro AUC: {roc_auc['micro']:.6f}\nMacro AUC: {roc_auc['macro']:.6f}\n")

    # t-SNE (2D) on full scaled features
    n_samples = X_scaled.shape[0]
    safe_perp = max(5, min(30, (n_samples // 3) - 1)) if n_samples > 50 else 5
    tsne = make_tsne(perplexity=safe_perp)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    enc_all = le.transform(features_df["Label"].values)
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    for idx, name in enumerate(class_names):
        mask = (enc_all == idx)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    marker=markers[idx % len(markers)],
                    label=name, s=60, linewidths=0.5)
    plt.xlabel("t-SNE 1", fontsize=16, fontweight='bold')
    plt.ylabel("t-SNE 2", fontsize=16, fontweight='bold')
    plt.legend(title="Class", title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    tsne_path = OUTDIR / "tsne.png"
    plt.savefig(tsne_path, dpi=300)
    plt.close()

    # Save t-SNE coords
    pd.DataFrame({"tsne_1": X_tsne[:, 0], "tsne_2": X_tsne[:, 1], "label": features_df["Label"].values}).to_csv(
        OUTDIR / "tsne_coordinates.csv", index=False
    )

    # Save feature importances (RF; XGB if available)
    try:
        rf_used = rf.best_estimator_ if isinstance(rf, GridSearchCV) else rf
        imp = getattr(rf_used, "feature_importances_", None)
        if imp is not None:
            cols = [c for c in features_df.columns if c != "Label"]
            fi = pd.DataFrame({"feature": cols, "rf_importance": imp}).sort_values("rf_importance", ascending=False)
            fi.to_csv(OUTDIR / "feature_importance_rf.csv", index=False)
    except Exception:
        pass
    if xgb_clf is not None:
        try:
            cols = [c for c in features_df.columns if c != "Label"]
            xgi = xgb_clf.get_booster().get_score(importance_type="gain")
            # align to columns (XGBoost names features as f0, f1, ...)
            rows = []
            for i, c in enumerate(cols):
                rows.append((c, xgi.get(f"f{i}", 0.0)))
            pd.DataFrame(rows, columns=["feature", "xgb_gain"]).sort_values("xgb_gain", ascending=False)\
                .to_csv(OUTDIR / "feature_importance_xgb.csv", index=False)
        except Exception:
            pass

    # Predictions CSV
    pred_df = pd.DataFrame({
        "y_true": [class_names[i] for i in y_test],
        "y_pred": [class_names[i] for i in y_pred]
    })
    pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)

    # Summary
    print("\n=== Saved Outputs ===")
    print(f"- Features Excel used: {features_path.resolve()}")
    print(f"- Confusion Matrix:    {cm_path.resolve()}")
    print(f"- ROC Curves:          {roc_path.resolve()}")
    print(f"- t-SNE Plot:          {tsne_path.resolve()}")
    print(f"- t-SNE CSV:           {(OUTDIR / 'tsne_coordinates.csv').resolve()}")
    print(f"- Report (txt):        {(OUTDIR / 'classification_report.txt').resolve()}")
    print(f"- ROC AUC (txt):       {(OUTDIR / 'roc_auc_report.txt').resolve()}")
    print(f"- Confusion CSV:       {(OUTDIR / 'confusion_matrix.csv').resolve()}")
    print(f"- Predictions CSV:     {(OUTDIR / 'test_predictions.csv').resolve()}")
    if Path(OUTDIR / "feature_importance_rf.csv").exists():
        print(f"- RF Importances:      {(OUTDIR / 'feature_importance_rf.csv').resolve()}")
    if Path(OUTDIR / "feature_importance_xgb.csv").exists():
        print(f"- XGB Importances:     {(OUTDIR / 'feature_importance_xgb.csv').resolve()}")

    elapsed = time.time() - start_time
    print(f"\n=== Done in {elapsed:.1f} s ===")

if __name__ == "__main__":
    main()
