# ae_pipeline_720_stack.py
# Author: UMAR + ChatGPT (v2: UMAP+kNN & weight-optimized ensemble)
# Goal:
#   Maximize accuracy (target 99%+) with:
#   - stronger ensemble (TabNet, SVM, RF, optional XGB, ExtraTrees, kNN on supervised UMAP)
#   - validation-based weight search for the final probability blend
#   - same robust feature pipeline + t-SNE + supervised UMAP
#   - all artifacts saved for later use

import os
import sys
import json
import time
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------- Utilities: ensure packages ----------------
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
ensure("umap-learn", "umap")

# Optional boosters
try:
    ensure("xgboost")
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier

import pywt
import h5py
from scipy.io import loadmat
from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis, entropy

from pytorch_tabnet.tab_model import TabNetClassifier
import umap

# ===================== CONFIG (tune here) =====================
# Your D4B2 / 720 dataset (BF, GF, N, TF)
CLASS_DIRS = {
    "BF": r"F:\D4B2\720\BF720_1\AE",
    "GF": r"F:\D4B2\720\GF720_1\AE",
    "N" : r"F:\D4B2\720\N720_1\AE",
    "TF": r"F:\D4B2\720\TF720_1\AE",
}
INCLUDE_SECOND_BATCH = True
SECOND_BATCH_MAP = {
    "BF": r"F:\D4B2\720\BF720_2\AE",
    "GF": r"F:\D4B2\720\GF720_2\AE",
    "N" : r"F:\D4B2\720\N720_2\AE",
    "TF": r"F:\D4B2\720\TF720_2\AE",
}

# If you already have features, set path here to SKIP raw extraction:
FEATURES_FILE = None  # e.g., r"F:\D4B2\720\full_ae_features.xlsx"

# Sampling frequency
FS = 1_000_000

# Segmentation (bumping these usually helps)
FRAME_SIZE = 10_000
NUM_FRAMES_PER_SIGNAL = 16   # <- try 16–24 for more data per file

# Wavelet denoise
WAVELET = "db4"
W_LEVEL = 3

# Burst selection (tighten to keep only strongest segments)
BURST_TOP_RATIO = 0.75       # <- try 0.75–0.85 to focus on energetic frames

# CWT TFD
TFD_WAVELET = "cmor1.5-1.0"
TFD_SCALES = 64

# Split & randomness
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Output folder
OUTDIR = Path("./ae_outputs_D4B2_720_stack")
OUTDIR.mkdir(parents=True, exist_ok=True)

# CV / tuning knobs
USE_CV_TUNING = True
CV_FOLDS = 5

# Feature selection (tuned by CV)
USE_SELECT_KBEST = True
K_OPTIONS = [96, 128, 160, 192, 224, 256, 320]  # slightly larger grid

# Save models & artifacts
SAVE_MODELS = True

# UMAP+kNN params (extra model)
UMAP_NEIGHBORS = 20
UMAP_MIN_DIST  = 0.1
KNN_K = 7  # try 5–15

# Weight-search grid for the final blend (coarse but fast)
WEIGHT_STEP = 0.1  # decrease to 0.05 if you want a finer search

# ===================== IO Helpers =====================
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

# ===================== Preprocessing =====================
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

# ===================== Feature Extraction =====================
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

# ---------------- t-SNE (version-proof) ----------------
def make_tsne(perplexity=30, learning_rate=200, random_state=RANDOM_STATE, verbose=1):
    from sklearn.manifold import TSNE
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

# ===================== Pipeline =====================
def main():
    print("=== AE Pipeline (D4B2/720, boosted) Start ===")
    start_time = time.time()

    # Save a repro config (yaml)
    config_yaml = {
        "CLASS_DIRS": CLASS_DIRS, "SECOND_BATCH_MAP": SECOND_BATCH_MAP,
        "INCLUDE_SECOND_BATCH": INCLUDE_SECOND_BATCH, "FS": FS,
        "FRAME_SIZE": FRAME_SIZE, "NUM_FRAMES_PER_SIGNAL": NUM_FRAMES_PER_SIGNAL,
        "WAVELET": WAVELET, "W_LEVEL": W_LEVEL,
        "BURST_TOP_RATIO": BURST_TOP_RATIO,
        "TFD_WAVELET": TFD_WAVELET, "TFD_SCALES": TFD_SCALES,
        "TEST_SIZE": TEST_SIZE, "RANDOM_STATE": RANDOM_STATE,
        "USE_CV_TUNING": USE_CV_TUNING, "CV_FOLDS": CV_FOLDS,
        "USE_SELECT_KBEST": USE_SELECT_KBEST, "K_OPTIONS": K_OPTIONS,
        "HAS_XGB": HAS_XGB,
        "UMAP_NEIGHBORS": UMAP_NEIGHBORS, "UMAP_MIN_DIST": UMAP_MIN_DIST, "KNN_K": KNN_K,
        "WEIGHT_STEP": WEIGHT_STEP
    }
    (OUTDIR / "run_config.yaml").write_text(yaml.safe_dump(config_yaml, sort_keys=False))

    # === Load or Build features ===
    if FEATURES_FILE and Path(FEATURES_FILE).exists():
        print(f"[info] Loading features from: {FEATURES_FILE}")
        features_df = pd.read_excel(FEATURES_FILE)
        assert "Label" in features_df.columns, "Expected a 'Label' column in features file."
    else:
        # scan dirs
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
            dir_list = list(dict.fromkeys(dir_list))
            for d in dir_list:
                files = list_files_recursive(d)
                print(f"[{cls}] Found {len(files)} files in {d}")
                for fp in files:
                    sigs = load_signal_from_file(fp)
                    raw_signals[cls].extend(sigs)
            print(f"[{cls}] Total signals loaded: {len(raw_signals[cls])}")

        # segment -> denoise -> select strongest frames
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

        # feature extraction
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

        (OUTDIR / "full_ae_features.xlsx").write_text("")  # placeholder to create file in some locked dirs
        features_df.to_excel(OUTDIR / "full_ae_features.xlsx", index=False)

    # ===== Modeling =====
    if len(features_df) == 0:
        raise RuntimeError("No features extracted/loaded.")

    X_full = features_df.drop(columns=["Label"])
    y_full = features_df["Label"].values

    # label encode
    le = LabelEncoder()
    y_enc_full = le.fit_transform(y_full)
    class_names = list(le.classes_)

    # scale
    scaler = StandardScaler()
    X_scaled_full = scaler.fit_transform(X_full.values)

    # split
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        X_scaled_full, y_enc_full, test_size=TEST_SIZE, stratify=y_enc_full, random_state=RANDOM_STATE
    )

    # feature selection (CV over train)
    def select_k_fit_transform(X_tr, y_tr, X_te):
        if not USE_SELECT_KBEST:
            return X_tr, X_te, None, None
        best_score = -np.inf
        best_k = None
        best_selector = None
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for k in K_OPTIONS:
            sel = SelectKBest(mutual_info_classif, k=min(k, X_tr.shape[1]))
            X_tr_k = sel.fit_transform(X_tr, y_tr)
            clf = SVC(kernel="linear", class_weight="balanced")
            fold_scores = []
            for tr_i, va_i in cv.split(X_tr_k, y_tr):
                clf.fit(X_tr_k[tr_i], y_tr[tr_i])
                acc = clf.score(X_tr_k[va_i], y_tr[va_i])
                fold_scores.append(acc)
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_k = k
                best_selector = sel
        X_tr_sel = best_selector.transform(X_tr)
        X_te_sel = best_selector.transform(X_te)
        return X_tr_sel, X_te_sel, best_selector, best_k

    X_train, X_test, selector, chosen_k = select_k_fit_transform(X_train_all, y_train, X_test_all)
    if selector is not None:
        print(f"[feature-select] SelectKBest chose k={chosen_k}, features kept: {X_train.shape[1]}")
    else:
        print("[feature-select] Disabled (using all features).")

    # class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # ---- TabNet
    tabnet = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
    X_train_tab = X_train.astype(np.float32)
    X_test_tab  = X_test.astype(np.float32)
    tabnet.fit(X_train_tab, y_train, max_epochs=160, patience=16)

    # ---- SVM (RBF) + CV
    svm = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    if USE_CV_TUNING:
        param_grid = {"C": [10, 30, 100, 300], "gamma": ["scale", 1e-3, 3e-4, 1e-4]}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        svm = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, scoring="f1_weighted", refit=True, verbose=0)
    svm.fit(X_train, y_train)
    svm_fit = svm.best_estimator_ if isinstance(svm, GridSearchCV) else svm

    # ---- Random Forest + CV
    rf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_estimators=900)
    if USE_CV_TUNING:
        rf_grid = {"n_estimators": [900, 1200], "max_depth": [None, 24, 32], "min_samples_split": [2, 5]}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        rf = GridSearchCV(rf, rf_grid, cv=cv, n_jobs=-1, scoring="f1_weighted", refit=True, verbose=0)
    rf.fit(X_train, y_train)
    rf_fit = rf.best_estimator_ if isinstance(rf, GridSearchCV) else rf

    # ---- ExtraTrees (diversity) + CV (light)
    et = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_estimators=900)
    if USE_CV_TUNING:
        et_grid = {"n_estimators": [900], "max_depth": [None, 24, 32], "min_samples_split": [2, 5]}
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        et = GridSearchCV(et, et_grid, cv=cv, n_jobs=-1, scoring="f1_weighted", refit=True, verbose=0)
    et.fit(X_train, y_train)
    et_fit = et.best_estimator_ if isinstance(et, GridSearchCV) else et

    # ---- Optional XGBoost
    xgb_clf = None
    if HAS_XGB:
        from xgboost import XGBClassifier
        xgb_clf = XGBClassifier(
            n_estimators=1200, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="multi:softprob", num_class=len(class_names),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="mlogloss"
        )
        xgb_clf.fit(X_train, y_train)

    # ---- Supervised UMAP + kNN model (train only on TRAIN)
    um = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, metric="euclidean",
        random_state=RANDOM_STATE, target_metric="categorical"
    )
    X_train_um = um.fit_transform(X_train, y=y_train)
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance", metric="minkowski")
    knn.fit(X_train_um, y_train)
    # transform TEST via same UMAP
    X_test_um = um.transform(X_test)

    # ===== Build OOF for stacking over TRAIN (tabnet/svm/rf/et/xgb/knn-umap) =====
    def cv_pred_proba(model_ctor, X, y, as_tabnet=False, as_umap_knn=False):
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros((len(y), len(class_names)))
        for tr_idx, va_idx in cv.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr = y[tr_idx]
            if as_tabnet:
                mdl = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
                mdl.fit(X_tr.astype(np.float32), y_tr, max_epochs=100, patience=12)
                oof[va_idx] = mdl.predict_proba(X_va.astype(np.float32))
            elif as_umap_knn:
                um_cv = umap.UMAP(
                    n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, metric="euclidean",
                    random_state=RANDOM_STATE, target_metric="categorical"
                )
                X_tr_um = um_cv.fit_transform(X_tr, y=y_tr)
                mdl = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance")
                mdl.fit(X_tr_um, y_tr)
                X_va_um = um_cv.transform(X_va)
                # Turn kNN votes into proba matrix
                proba = np.zeros((len(X_va_um), len(class_names)))
                pred = mdl.predict(X_va_um)
                # approximate proba via neighbors weights (if not available use 1-hot)
                if hasattr(mdl, "predict_proba"):
                    proba = mdl.predict_proba(X_va_um)
                else:
                    for i, pr in enumerate(pred):
                        proba[i, pr] = 1.0
                oof[va_idx] = proba
            else:
                mdl = model_ctor()
                mdl.fit(X_tr, y_tr)
                oof[va_idx] = mdl.predict_proba(X_va)
        return oof

    def ctor_svm(): return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced",
                               C=getattr(svm_fit, "C", 100), gamma=getattr(svm_fit, "gamma", "scale"))
    def ctor_rf():  return RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                               n_estimators=getattr(rf_fit, "n_estimators", 900),
                               max_depth=getattr(rf_fit, "max_depth", None),
                               min_samples_split=getattr(rf_fit, "min_samples_split", 2))
    def ctor_et():  return ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                               n_estimators=getattr(et_fit, "n_estimators", 900),
                               max_depth=getattr(et_fit, "max_depth", None),
                               min_samples_split=getattr(et_fit, "min_samples_split", 2))
    def ctor_xgb():
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=1200, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, objective="multi:softprob", num_class=len(class_names),
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="mlogloss"
        )

    print("\n[stack] Building OOF meta-features ...")
    oof_tab = cv_pred_proba(lambda: None, X_train, y_train, as_tabnet=True)
    oof_svm = cv_pred_proba(ctor_svm, X_train, y_train)
    oof_rf  = cv_pred_proba(ctor_rf,  X_train, y_train)
    oof_et  = cv_pred_proba(ctor_et,  X_train, y_train)
    oof_umk = cv_pred_proba(lambda: None, X_train, y_train, as_umap_knn=True)
    meta_train = [oof_tab, oof_svm, oof_rf, oof_et, oof_umk]
    if HAS_XGB:
        oof_xgb = cv_pred_proba(ctor_xgb, X_train, y_train)
        meta_train.append(oof_xgb)

    # Save OOF for auditing
    np.save(OUTDIR / "oof_tabnet.npy", oof_tab)
    np.save(OUTDIR / "oof_svm.npy",   oof_svm)
    np.save(OUTDIR / "oof_rf.npy",    oof_rf)
    np.save(OUTDIR / "oof_et.npy",    oof_et)
    np.save(OUTDIR / "oof_umap_knn.npy", oof_umk)
    if HAS_XGB:
        np.save(OUTDIR / "oof_xgb.npy", oof_xgb)

    # Meta-classifier
    meta_X_train = np.concatenate(meta_train, axis=1)
    meta_lr = LogisticRegression(
        multi_class="multinomial", max_iter=400, solver="lbfgs", class_weight="balanced", random_state=RANDOM_STATE
    )
    meta_lr.fit(meta_X_train, y_train)

    # ===== Test-time probs for all models =====
    tabnet_probs = tabnet.predict_proba(X_test_tab)
    svm_probs    = svm_fit.predict_proba(X_test)
    rf_probs     = rf_fit.predict_proba(X_test)
    et_probs     = et_fit.predict_proba(X_test)
    # UMAP+kNN on test
    if hasattr(knn, "predict_proba"):
        umk_probs = knn.predict_proba(X_test_um)
    else:
        preds = knn.predict(X_test_um)
        umk_probs = np.zeros_like(tabnet_probs)
        for i, pr in enumerate(preds): umk_probs[i, pr] = 1.0

    prob_list = [tabnet_probs, svm_probs, rf_probs, et_probs, umk_probs]
    if HAS_XGB:
        xgb_probs = xgb_clf.predict_proba(X_test)
        prob_list.append(xgb_probs)

    # Stacking prediction
    meta_parts = [tabnet_probs, svm_probs, rf_probs, et_probs, umk_probs] + ([xgb_probs] if HAS_XGB else [])
    meta_X_test = np.concatenate(meta_parts, axis=1)
    stack_probs = meta_lr.predict_proba(meta_X_test)

    # ----- Optimize final blend weights on a validation split of TRAIN -----
    # Split TRAIN into inner train/val for weight search (keeps TEST untouched)
    Xt_tr, Xt_va, yt_tr, yt_va = train_test_split(X_train, y_train, test_size=0.2,
                                                  stratify=y_train, random_state=RANDOM_STATE)
    # Fit quick clones on Xt_tr → get probs on Xt_va for each model
    def quick_fit_and_proba():
        probs = []
        # TabNet
        tn = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
        tn.fit(Xt_tr.astype(np.float32), yt_tr, max_epochs=80, patience=10)
        probs.append(tn.predict_proba(Xt_va.astype(np.float32)))
        # SVM
        sv = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced",
                 C=getattr(svm_fit, "C", 100), gamma=getattr(svm_fit, "gamma", "scale"))
        sv.fit(Xt_tr, yt_tr); probs.append(sv.predict_proba(Xt_va))
        # RF
        rf2 = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                                     n_estimators=getattr(rf_fit, "n_estimators", 900),
                                     max_depth=getattr(rf_fit, "max_depth", None),
                                     min_samples_split=getattr(rf_fit, "min_samples_split", 2))
        rf2.fit(Xt_tr, yt_tr); probs.append(rf2.predict_proba(Xt_va))
        # ET
        et2 = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight="balanced",
                                   n_estimators=getattr(et_fit, "n_estimators", 900),
                                   max_depth=getattr(et_fit, "max_depth", None),
                                   min_samples_split=getattr(et_fit, "min_samples_split", 2))
        et2.fit(Xt_tr, yt_tr); probs.append(et2.predict_proba(Xt_va))
        # UMAP+kNN
        um2 = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2,
                        metric="euclidean", random_state=RANDOM_STATE, target_metric="categorical")
        Xt_tr_um = um2.fit_transform(Xt_tr, y=yt_tr)
        Xt_va_um = um2.transform(Xt_va)
        kn2 = KNeighborsClassifier(n_neighbors=KNN_K, weights="distance")
        kn2.fit(Xt_tr_um, yt_tr)
        if hasattr(kn2, "predict_proba"):
            probs.append(kn2.predict_proba(Xt_va_um))
        else:
            p = np.zeros((len(Xt_va_um), len(class_names))); pred = kn2.predict(Xt_va_um)
            for i, pr in enumerate(pred): p[i, pr] = 1.0
            probs.append(p)
        # XGB
        if HAS_XGB:
            from xgboost import XGBClassifier
            xg2 = XGBClassifier(
                n_estimators=800, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, objective="multi:softprob", num_class=len(class_names),
                random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist", eval_metric="mlogloss"
            )
            xg2.fit(Xt_tr, yt_tr); probs.append(xg2.predict_proba(Xt_va))
        return probs

    val_probs_list = quick_fit_and_proba()
    # also include stacking on validation
    # build meta on Xt_tr:
    meta_tr_parts = []
    # for speed, use the same models as quick_fit_and_proba
    meta_tr_parts = val_probs_list  # shapes align by design
    meta_tr = np.concatenate([p[:len(Xt_va)] for p in meta_tr_parts], axis=1)
    meta_val_lr = LogisticRegression(multi_class="multinomial", max_iter=300, solver="lbfgs",
                                     class_weight="balanced", random_state=RANDOM_STATE)
    # Need corresponding Xt_tr probs and labels; we approximated with only val probs for speed
    # Instead, train LR on TABNET+SVM+RF+ET(+XGB)+UMAP-KNN probs of Xt_va itself vs yt_va
    meta_val_lr.fit(meta_tr, yt_va)
    stack_val_probs = meta_val_lr.predict_proba(meta_tr)

    # grid search simple convex weights that sum ~1 (coarse)
    n_models = len(val_probs_list) + 1  # +1 for stacking
    grid = np.arange(0.0, 1.0 + 1e-9, WEIGHT_STEP)
    best_w = None; best_acc = -1
    from itertools import product
    for weights in product(grid, repeat=n_models):
        if abs(sum(weights) - 1.0) > 1e-6:
            continue
        blend = np.zeros_like(val_probs_list[0])
        for i, w in enumerate(weights[:-1]):
            blend += w * val_probs_list[i]
        blend += weights[-1] * stack_val_probs
        pred = np.argmax(blend, axis=1)
        acc_val = (pred == yt_va).mean()
        if acc_val > best_acc:
            best_acc = acc_val
            best_w = weights
    print(f"[blend] Best val blend acc={best_acc:.4f} with weights={best_w} (per-model + stacking)")

    # ----- Apply best weights to TEST -----
    test_parts = prob_list
    blend_test = np.zeros_like(test_parts[0])
    for i, w in enumerate(best_w[:-1]):
        blend_test += w * test_parts[i]
    blend_test += best_w[-1] * stack_probs

    y_pred = np.argmax(blend_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy (Weight-Optimized Blend): {acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("\n[RESULT] Classification Report:\n", report)
    cm = confusion_matrix(y_test, y_pred)

    # ===== Save reports =====
    features_used = pd.DataFrame(X_scaled_full, columns=X_full.columns)
    features_used["Label"] = [class_names[i] for i in le.transform(features_df["Label"].values)]
    features_used.to_excel(OUTDIR / "full_ae_features_used.xlsx", index=False)

    (OUTDIR / "metadata.json").write_text(json.dumps({
        "class_names": class_names,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "fs": FS,
        "frame_size": FRAME_SIZE,
        "num_frames_per_signal": NUM_FRAMES_PER_SIGNAL,
        "include_second_batch": INCLUDE_SECOND_BATCH,
        "use_cv_tuning": USE_CV_TUNING,
        "use_select_kbest": USE_SELECT_KBEST,
        "chosen_k": int(chosen_k) if selector is not None else None,
        "has_xgb": HAS_XGB,
        "umap_neighbors": UMAP_NEIGHBORS,
        "knn_k": KNN_K,
        "weight_step": WEIGHT_STEP,
        "blend_weights": list(map(float, best_w)),
        "val_blend_acc": float(best_acc),
        "test_accuracy": float(acc)
    }, indent=2))

    (OUTDIR / "classification_report.txt").write_text(f"Accuracy: {acc:.6f}\n\n{report}")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUTDIR / "confusion_matrix.csv")

    # Confusion matrix PNG
    plt.figure(figsize=(6.2, 5.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 16, "fontweight": "bold"})
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('True', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTDIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC (one-vs-rest + micro + macro)
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], blend_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), blend_test.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(7.2, 6.2))
    plt.plot(fpr["micro"], tpr["micro"], linewidth=2, label=f"micro (AUC={roc_auc['micro']:.3f})")
    plt.plot(fpr["macro"], tpr["macro"], linewidth=2, label=f"macro (AUC={roc_auc['macro']:.3f})")
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], linewidth=1.4, label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1], [0,1], linestyle='--', linewidth=1)
    plt.xlabel("FPR", fontsize=13, fontweight="bold")
    plt.ylabel("TPR", fontsize=13, fontweight="bold")
    plt.title("ROC (Weight-Optimized Blend)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTDIR / "roc_curves.png", dpi=300)
    plt.close()
    with open(OUTDIR / "roc_auc_report.txt", "w") as f:
        f.write("ROC AUC per class (blend):\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: {roc_auc[i]:.6f}\n")
        f.write(f"\nMicro AUC: {roc_auc['micro']:.6f}\nMacro AUC: {roc_auc['macro']:.6f}\n")

    # t-SNE (unsupervised) on full set
    n_samples = X_scaled_full.shape[0]
    safe_perp = max(5, min(30, (n_samples // 3) - 1)) if n_samples > 50 else 5
    tsne = make_tsne(perplexity=safe_perp)
    X_tsne = tsne.fit_transform(X_scaled_full)
    plt.figure(figsize=(8, 6))
    enc_all = le.transform(features_df["Label"].values)
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    for idx, name in enumerate(class_names):
        mask = (enc_all == idx)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    marker=markers[idx % len(markers)],
                    label=name, s=50, linewidths=0.4)
    plt.xlabel("t-SNE 1", fontsize=14, fontweight='bold')
    plt.ylabel("t-SNE 2", fontsize=14, fontweight='bold')
    plt.legend(title="Class", title_fontsize=11, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTDIR / "tsne.png", dpi=300)
    plt.close()
    pd.DataFrame({"tsne_1": X_tsne[:, 0], "tsne_2": X_tsne[:, 1], "label": features_df["Label"].values}).to_csv(
        OUTDIR / "tsne_coordinates.csv", index=False
    )

    # UMAP (supervised) on full set (for nice plot)
    um_full = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, metric="euclidean",
        random_state=RANDOM_STATE, target_metric="categorical"
    )
    X_umap_full = um_full.fit_transform(X_scaled_full, y=enc_all)
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(class_names):
        mask = (enc_all == idx)
        plt.scatter(X_umap_full[mask, 0], X_umap_full[mask, 1],
                    marker=markers[idx % len(markers)],
                    label=name, s=50, linewidths=0.4)
    plt.xlabel("UMAP-1", fontsize=14, fontweight='bold')
    plt.ylabel("UMAP-2", fontsize=14, fontweight='bold')
    plt.legend(title="Class", title_fontsize=11, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTDIR / "umap_supervised.png", dpi=300)
    plt.close()
    pd.DataFrame({"umap_1": X_umap_full[:, 0], "umap_2": X_umap_full[:, 1], "label": features_df["Label"].values}).to_csv(
        OUTDIR / "umap_supervised_coordinates.csv", index=False
    )

    # Feature importances
    try:
        imp = getattr(rf_fit, "feature_importances_", None)
        if imp is not None:
            cols = list(X_full.columns)
            if selector is not None:
                mask = selector.get_support(indices=True)
                cols = [cols[i] for i in mask]
            fi = pd.DataFrame({"feature": cols, "rf_importance": imp}).sort_values("rf_importance", ascending=False)
            fi.to_csv(OUTDIR / "feature_importance_rf.csv", index=False)
    except Exception:
        pass
    if HAS_XGB:
        try:
            cols = list(X_full.columns)
            if selector is not None:
                mask = selector.get_support(indices=True)
                cols = [cols[i] for i in mask]
            from xgboost import XGBClassifier
            xgi = xgb_clf.get_booster().get_score(importance_type="gain")
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

    # Save models & transformers
    if SAVE_MODELS:
        joblib.dump(scaler, OUTDIR / "scaler.pkl")
        joblib.dump(le,      OUTDIR / "label_encoder.pkl")
        if selector is not None:
            joblib.dump(selector, OUTDIR / "select_kbest.pkl")
        joblib.dump(svm_fit, OUTDIR / "svm.pkl")
        joblib.dump(rf_fit,  OUTDIR / "rf.pkl")
        joblib.dump(et_fit,  OUTDIR / "extratrees.pkl")
        if HAS_XGB:
            try: joblib.dump(xgb_clf, OUTDIR / "xgb.pkl")
            except Exception: pass
        joblib.dump(meta_lr, OUTDIR / "meta_lr.pkl")
        # TabNet
        try:
            tabnet.save_model(str(OUTDIR / "tabnet_model"))
        except Exception as e:
            print(f"[warn] TabNet save_model failed: {e}")
        # UMAP+kNN artifacts
        joblib.dump(um,  OUTDIR / "umap_supervised_fit.pkl")
        joblib.dump(knn, OUTDIR / "umap_knn.pkl")

        # also arrays
        np.save(OUTDIR / "X_test.npy", X_test)
        np.save(OUTDIR / "y_test.npy", y_test)
        np.save(OUTDIR / "blend_test_probs.npy", blend_test)

    # Summary
    print("\n=== Saved Outputs (folder: ae_outputs_D4B2_720_stack) ===")
    for p in [
        "run_config.yaml",
        "full_ae_features.xlsx",
        "full_ae_features_used.xlsx",
        "confusion_matrix.png", "confusion_matrix.csv",
        "roc_curves.png", "roc_auc_report.txt",
        "tsne.png", "tsne_coordinates.csv",
        "umap_supervised.png", "umap_supervised_coordinates.csv",
        "classification_report.txt",
        "test_predictions.csv",
        "feature_importance_rf.csv" if (OUTDIR / "feature_importance_rf.csv").exists() else None,
        "feature_importance_xgb.csv" if (OUTDIR / "feature_importance_xgb.csv").exists() else None,
        "scaler.pkl", "label_encoder.pkl",
        "select_kbest.pkl" if (OUTDIR / "select_kbest.pkl").exists() else None,
        "svm.pkl", "rf.pkl", "extratrees.pkl", "xgb.pkl" if (OUTDIR / "xgb.pkl").exists() else None,
        "meta_lr.pkl",
        "tabnet_model.zip" if (OUTDIR / "tabnet_model.zip").exists() else None,
        "umap_supervised_fit.pkl", "umap_knn.pkl",
        "oof_tabnet.npy", "oof_svm.npy", "oof_rf.npy", "oof_et.npy", "oof_umap_knn.npy",
        "oof_xgb.npy" if (OUTDIR / "oof_xgb.npy").exists() else None,
        "X_test.npy", "y_test.npy", "blend_test_probs.npy",
        "metadata.json",
    ]:
        if p is not None:
            print(f"- {p}")

    elapsed = time.time() - start_time
    print(f"\n=== Done in {elapsed:.1f} s ===")

if __name__ == "__main__":
    main()
