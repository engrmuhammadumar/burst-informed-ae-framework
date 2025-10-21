# ae.py
# Author: UMAR + ChatGPT (stacked ensemble + paper-ready plots, no titles)
# End-to-end AE pipeline with robust IO, fast/parallel features, checkpoint/resume,
# stronger ensemble (TabNet + SVM + RF + XGBoost -> LR meta-learner),
# ROC/confusion/TSNE/UMAP (no titles), and NumPy 2.x trapz fix.

import os
import sys
import json
import time
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

# =========================
# Package bootstrap: ensure
# =========================
def ensure(package, import_name=None, extras=None):
    try:
        __import__(import_name or package)
    except ImportError:
        pkg = f"{package}[{extras}]" if extras else package
        print(f"[setup] Installing: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])
        __import__(import_name or package)

# Core stack
ensure("numpy")
ensure("scipy", "scipy")
ensure("pandas")
ensure("matplotlib", "matplotlib")
ensure("seaborn")
ensure("scikit-learn", "sklearn")
ensure("PyWavelets", "pywt")
ensure("h5py")
ensure("pytorch_tabnet", "pytorch_tabnet.tab_model")
ensure("joblib")
ensure("tqdm")
ensure("umap-learn", "umap")
ensure("xgboost")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
import h5py
import pywt
from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis, entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from pytorch_tabnet.tab_model import TabNetClassifier
from xgboost import XGBClassifier
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm
from umap import UMAP

# Silence numpy deprecation warnings just in case:
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")

# ---- Compatibility helper for trapezoidal integral (NumPy 1.x & 2.x) ----
def trapz_abs(frame):
    a = np.abs(frame)
    # NumPy 2.x
    if hasattr(np, "trapezoid"):
        return np.trapezoid(a)
    # NumPy 1.x fallback
    return np.trapz(a)

# ========== CONFIG ==========
# Your folders (both _1 and _2 included)
CLASS_DIRS = {
    "BF": [r"F:\D8B2\BF1440_1\AE", r"F:\D8B2\BF1440_2\AE"],
    "GF": [r"F:\D8B2\GF1440_1\AE", r"F:\D8B2\GF1440_2\AE"],
    "TF": [r"F:\D8B2\TF1440_1\AE", r"F:\D8B2\TF1440_2\AE"],
    "N" : [r"F:\D8B2\N1440_1\AE",  r"F:\D8B2\N1440_2\AE"],
}

# Sampling frequency
FS = 1_000_000   # 1e6

# Segmentation
FRAME_SIZE = 10_000
NUM_FRAMES_PER_SIGNAL = 10

# Wavelet denoise
WAVELET = "db4"
W_LEVEL = 3

# Burst selection: keep top 50% frames by energy
BURST_TOP_RATIO = 0.5

# FAST mode for CWT:
FAST_MODE = True                  # set False for full quality
CWT_DOWNSAMPLE = 2 if FAST_MODE else 1
CWT_SCALES = 32 if FAST_MODE else 64
TFD_WAVELET = "cmor1.5-1.0"

# Train/test
TEST_SIZE = 0.30
RANDOM_STATE = 42

# Parallel + checkpoint
N_JOBS = max(1, cpu_count() - 1)
CHECKPOINT_EVERY = 500

# Output paths
OUTDIR = Path("./ae_outputs"); OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT_CSV = OUTDIR / "features_checkpoint.csv"
FINAL_CSV = OUTDIR / "features_final.csv"      # full features (unscaled)
FEATURES_XLSX = OUTDIR / "full_ae_features.xlsx"  # scaled + labels for convenience

# ========= File discovery + loading =========
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
    """
    Robust loader supporting .npy, .csv, .txt, .mat (v7 + v7.3).
    Returns list of 1D numpy arrays (signals). Cleans NaNs/Infs.
    """
    p = Path(path)
    ext = p.suffix.lower()
    sigs = []
    try:
        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            arr = np.array(arr)
        elif ext in (".csv", ".txt"):
            arr = np.loadtxt(path, dtype=float, delimiter="," if ext == ".csv" else None)
        elif ext == ".mat":
            try:
                md = loadmat(path)
                candidates = []
                for k, v in md.items():
                    if k.startswith("__"):
                        continue
                    a = np.array(v)
                    if a.ndim in (1, 2):
                        candidates.append((k, a.size, a.shape))
                if not candidates:
                    raise ValueError("No 1D/2D arrays in MAT (v7).")
                candidates.sort(key=lambda x: x[1], reverse=True)
                key = candidates[0][0]
                arr = np.array(md[key]).squeeze()
            except Exception:
                with h5py.File(path, "r") as f:
                    def walk(h, prefix=""):
                        for k in h.keys():
                            obj = h[k]
                            name = f"{prefix}/{k}".strip("/")
                            if isinstance(obj, h5py.Dataset):
                                yield name, obj
                            elif isinstance(obj, h5py.Group):
                                yield from walk(obj, name)
                    best = None
                    for name, ds in walk(f):
                        if best is None or ds.size > best[1].size:
                            best = (name, ds)
                    if best is None:
                        raise ValueError("No datasets in MAT v7.3")
                    arr = np.array(best[1])
        else:
            print(f"[warn] Unsupported extension: {ext} ({path})")
            return []

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
    except Exception as e:
        print(f"[error] load failed: {path} -> {e}")
        return []

    cleaned = []
    for s in sigs:
        s = np.nan_to_num(np.array(s, dtype=float).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        cleaned.append(s)
    return cleaned

# ========= Preprocessing =========
def segment_signal(signal, frame_size=FRAME_SIZE, num_frames=NUM_FRAMES_PER_SIGNAL):
    segs = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if end <= len(signal):
            segs.append(signal[start:end])
    return segs

def denoise_wavelet(frame, wavelet=WAVELET, level=W_LEVEL):
    coeffs = pywt.wavedec(frame, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(frame)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    den = pywt.waverec(coeffs_thresh, wavelet=wavelet)
    return den[:len(frame)]

def compute_energy(frame):
    return np.sum(frame**2)

def select_burst_frames(frames, top_ratio=BURST_TOP_RATIO):
    energies = np.array([compute_energy(f) for f in frames])
    thr = np.percentile(energies, 100 * (1 - top_ratio))
    selected = [f for f, e in zip(frames, energies) if e >= thr]
    return selected, energies, thr

# ========= Features =========
def extract_td_fd_features(frame, fs=FS):
    features = []
    mean_val = np.mean(frame)
    std_val  = np.std(frame)
    var_val  = np.var(frame)
    rms_val  = np.sqrt(np.mean(frame**2))
    peak_val = np.max(np.abs(frame))
    ptp_val  = np.ptp(frame)
    crest_factor = peak_val / rms_val if rms_val != 0 else 0
    mav = np.mean(np.abs(frame))
    impulse_factor = peak_val / mav if mav != 0 else 0
    shape_factor   = rms_val / mav if mav != 0 else 0
    margin_factor  = peak_val / (np.mean(np.sqrt(np.abs(frame)))**2) if np.mean(np.sqrt(np.abs(frame))) != 0 else 0
    skewness = skew(frame)
    kurtv    = kurtosis(frame)
    energy   = np.sum(frame**2)
    zcr      = ((frame[:-1] * frame[1:]) < 0).sum()
    max_val  = np.max(frame)
    min_val  = np.min(frame)
    range_val = max_val - min_val
    duration  = len(frame) / fs
    envelope_area = trapz_abs(frame)  # NumPy 2.x safe

    freqs, psd = welch(frame, fs=fs, nperseg=1024)
    psd = np.nan_to_num(psd)
    spectral_energy   = np.sum(psd)
    spectral_centroid = (np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) != 0 else 0
    spectral_entropy  = entropy(psd)
    spectral_kurtosis = kurtosis(psd)
    spectral_skewness = skew(psd)
    spectral_flatness = (np.exp(np.mean(np.log(psd + 1e-12))) / np.mean(psd)) if np.mean(psd) != 0 else 0
    dominant_freq = freqs[np.argmax(psd)]
    cumsum_psd = np.cumsum(psd)
    median_freq = freqs[np.where(cumsum_psd >= np.sum(psd) / 2)[0][0]]
    mean_freq   = spectral_centroid
    freq_variance = np.var(psd)
    freq_rms     = np.sqrt(np.mean(psd**2))
    freq_spread  = np.std(freqs)
    harmonics, _ = find_peaks(psd, height=np.max(psd)*0.1)
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

def extract_tfd_features_cwt_fast(frame, wavelet=TFD_WAVELET, fs=FS, num_scales=CWT_SCALES, ds=CWT_DOWNSAMPLE):
    if ds > 1:
        frame = frame[::ds]; fs = fs / ds
    scales = np.arange(1, num_scales + 1)
    coef, _ = pywt.cwt(frame, scales=scales, wavelet=wavelet, sampling_period=1/fs)
    scalogram = np.abs(coef)
    flat = scalogram.ravel()
    tfd_energy = np.sum(scalogram**2)
    tfd_entropy = entropy(flat)
    tfd_mean = np.mean(scalogram)
    tfd_std  = np.std(scalogram)
    tfd_max  = np.max(scalogram)
    tfd_kurt = kurtosis(flat)
    tfd_skew = skew(flat)
    tfd_centroid = np.sum(scalogram * np.arange(scalogram.shape[0])[:, None]) / (np.sum(scalogram) + 1e-12)
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
    bispec_mean = float(np.mean(bispec))
    bicoherence_mean = float(np.mean(bicoherence))
    nonlinearity_index = float(c3**2 + c4**2)
    return [m3, m4, m5, c3, c4, g_index, bispec_mean, bicoherence_mean, nonlinearity_index]

def extract_burst_features(frame, fs=FS, threshold_ratio=0.25):
    threshold = threshold_ratio * np.max(np.abs(frame))
    idx = np.where(np.abs(frame) > threshold)[0]
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
    peaks   = [np.max(np.abs(frame[s:s+int(d*fs)])) for s, d in zip(starts, durations)]
    rmss    = [np.sqrt(np.mean(frame[s:s+int(d*fs)]**2)) for s, d in zip(starts, durations)]
    ibis    = np.diff(starts) / fs if len(starts) > 1 else np.array([])
    total_event_duration = float(np.sum(durations))
    cumulative_counts = int(np.sum([int(d*fs) for d in durations]))
    return [
        len(durations), float(np.mean(durations)), float(np.mean(ibis)) if len(ibis) else 0.0,
        float(np.mean(energies)), float(np.mean(peaks)), float(np.mean(rmss)),
        cumulative_counts, total_event_duration,
        float(np.max(peaks)), float(np.min(peaks))
    ]

# ========= Extraction worker =========
TD_FD_NAMES = [
    "Mean","RMS","STD","Variance","Skewness","Kurtosis",
    "Peak","PeakToPeak","CrestFactor","ImpulseFactor","ShapeFactor","MarginFactor",
    "SignalEnergy","MaxVal","MinVal","Range","ZCR","Duration","MAV","EnvelopeArea",
    "MeanFreq","MedianFreq","DominantFreq","SpectralCentroid","SpectralEntropy",
    "SpectralKurtosis","SpectralSkewness","SpectralFlatness","SpectralEnergy",
    "FreqVariance","FreqRMS","FreqSpread","THD"
]
TFD_NAMES = [
    "TFD_Energy","TFD_Entropy","TFD_Mean","TFD_STD","TFD_Max",
    "TFD_Kurtosis","TFD_Skewness","TFD_Centroid","TFD_RidgeEnergyRatio"
]
HOS_NAMES = [
    "Moment3","Moment4","Moment5","Cumulant3","Cumulant4",
    "GaussianityIndex","BispectrumMean","BicoherenceMean","NonlinearityIndex"
]
BURST_NAMES = [
    "NumBursts","AvgBurstDuration","AvgInterBurstInterval","AvgBurstEnergy","AvgBurstPeak",
    "AvgBurstRMS","CumulativeCounts","TotalEventDuration","MaxBurstPeak","MinBurstPeak"
]
ALL_COLS = TD_FD_NAMES + TFD_NAMES + HOS_NAMES + BURST_NAMES + ["Label"]

def extract_all_features_for_frame(frame, label):
    td  = extract_td_fd_features(frame)
    tfd = extract_tfd_features_cwt_fast(frame)
    hos = extract_hos_features(frame)
    bf  = extract_burst_features(frame)
    return td + tfd + hos + bf + [label]

# ========= Helper: version-safe TSNE =========
def make_tsne(random_state=RANDOM_STATE, **kw):
    try:
        return TSNE(n_components=2, random_state=random_state, **kw)
    except TypeError:
        kw.pop("learning_rate", None)
        kw.pop("n_iter", None)
        kw.pop("init", None)
        return TSNE(n_components=2, random_state=random_state, **kw)

# ========= Main =========
def main():
    print("=== AE Pipeline Start ===")

    # 1) Load signals
    raw_signals = {k: [] for k in CLASS_DIRS.keys()}
    for cls, dir_list in CLASS_DIRS.items():
        for d in dir_list:
            files = list_files_recursive(d)
            print(f"[{cls}] Found {len(files)} files in {d}")
            for fp in files:
                sigs = load_signal_from_file(fp)
                raw_signals[cls].extend(sigs)
        print(f"[{cls}] Total signals loaded: {len(raw_signals[cls])}")

    # 2) Segment → denoise → burst select
    burst_selected_data = {}
    for cls, signals in raw_signals.items():
        frames_all = []
        for s in signals:
            segs = segment_signal(s, FRAME_SIZE, NUM_FRAMES_PER_SIGNAL)
            segs = [seg for seg in segs if len(seg) == FRAME_SIZE]
            segs = [denoise_wavelet(seg) for seg in segs]
            frames_all.extend(segs)
        selected, energies, thr = select_burst_frames(frames_all, BURST_TOP_RATIO)
        burst_selected_data[cls] = selected
        print(f"[{cls}] frames: {len(frames_all)} → selected: {len(selected)} (thr={thr:.2e})")

    # 3) Feature extraction with resume/checkpoint
    if FINAL_CSV.exists():
        print(f"[features] Found final features: {FINAL_CSV}. Loading to skip extraction.")
        features_df = pd.read_csv(FINAL_CSV)
    else:
        all_items = []
        for cls, frames in burst_selected_data.items():
            for f in frames:
                all_items.append((cls, f))
        total = len(all_items)
        print(f"[features] Total selected frames: {total}")
        td = []
        done_rows = 0
        if CKPT_CSV.exists():
            ckpt = pd.read_csv(CKPT_CSV)
            done_rows = len(ckpt)
            print(f"[features] Resuming from checkpoint with {done_rows} rows.")
            all_items = all_items[done_rows:]
            td.append(ckpt)

        cols = ALL_COLS
        for i in tqdm(range(0, len(all_items), CHECKPOINT_EVERY), desc="[features]"):
            chunk = all_items[i:i+CHECKPOINT_EVERY]
            results = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(extract_all_features_for_frame)(frame, label) for (label, frame) in chunk
            )
            chunk_df = pd.DataFrame(results, columns=cols)
            td.append(chunk_df)
            pd.concat(td, axis=0, ignore_index=True).to_csv(CKPT_CSV, index=False)

        features_df = pd.concat(td, axis=0, ignore_index=True)
        features_df.to_csv(FINAL_CSV, index=False)
        print(f"[features] Saved final features to {FINAL_CSV}")

    # 4) Scale + split
    labels = features_df["Label"].values
    X = features_df.drop(columns=["Label"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)

    # also export convenient Excel (scaled)
    scaled_df = pd.DataFrame(X_scaled, columns=features_df.drop(columns=["Label"]).columns)
    scaled_df["Label"] = labels
    scaled_df.to_excel(FEATURES_XLSX, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 5) Base models (stronger defaults)
    X_train_tab = X_train.astype(np.float32)
    X_test_tab  = X_test.astype(np.float32)

    tabnet = TabNetClassifier(verbose=0, seed=RANDOM_STATE)
    tabnet.fit(X_train_tab, y_train)

    svm = SVC(kernel="rbf", C=12, gamma="scale", probability=True,
              class_weight=None, random_state=RANDOM_STATE)
    svm.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0,
        objective="multi:softprob", num_class=len(class_names),
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    # 6) Probabilities (base learners)
    tabnet_probs = tabnet.predict_proba(X_test_tab)
    svm_probs    = svm.predict_proba(X_test)
    rf_probs     = rf.predict_proba(X_test)
    xgb_probs    = xgb.predict_proba(X_test)

    # 7) Stacked ensemble (meta-learner on validation set via simple split)
    # Build meta features on train (out-of-fold would be best, but this is fast & reliable)
    tab_train = tabnet.predict_proba(X_train_tab)
    svm_train = svm.predict_proba(X_train)
    rf_train  = rf.predict_proba(X_train)
    xgb_train = xgb.predict_proba(X_train)

    F_train = np.hstack([tab_train, svm_train, rf_train, xgb_train])
    F_test  = np.hstack([tabnet_probs, svm_probs, rf_probs, xgb_probs])

    meta = LogisticRegression(max_iter=200, multi_class="ovr", solver="liblinear", random_state=RANDOM_STATE)
    meta.fit(F_train, y_train)
    ens_probs = meta.predict_proba(F_test)

    # 8) Evaluation
    y_pred = np.argmax(ens_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy (Stacked Ensemble): {acc:.4f}\n")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print("[RESULT] Classification Report:\n", report)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUTDIR / "confusion_matrix.csv")
    (OUTDIR / "classification_report.txt").write_text(f"Accuracy: {acc:.6f}\n\n{report}")

    # 9) Styled confusion matrix (NO TITLE)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=False, annot_kws={"size": 18, "fontweight": "bold"})
    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTDIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 10) ROC curves (OvR + micro/macro) — NO TITLE
    y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], ens_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), ens_probs.ravel())
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
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUTDIR / "roc_curves.png", dpi=300)
    plt.close()

    with open(OUTDIR / "roc_auc_report.txt", "w") as f:
        f.write("ROC AUC per class (stacked ensemble):\n")
        for i, name in enumerate(class_names):
            f.write(f"{name}: {roc_auc[i]:.6f}\n")
        f.write(f"\nMicro AUC: {roc_auc['micro']:.6f}\nMacro AUC: {roc_auc['macro']:.6f}\n")

    # 11) t-SNE (NO TITLE)
    tsne = make_tsne(perplexity=50, learning_rate=300, n_iter=1500, init="pca")
    X_tsne = tsne.fit_transform(X_scaled)
    markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
    plt.figure(figsize=(8, 6))
    label_ids = le.transform(features_df["Label"].values)
    for idx, name in enumerate(class_names):
        mask = (label_ids == idx)
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                    marker=markers[idx % len(markers)],
                    label=name, s=60, linewidths=0.5)
    plt.xlabel("t-SNE 1", fontsize=16, fontweight='bold')
    plt.ylabel("t-SNE 2", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.legend(title="Class", title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTDIR / "tsne.png", dpi=300)
    plt.close()

    # 12) UMAP supervised (NO TITLE) — this is the clean separator you liked
    umap_sup = UMAP(n_components=2, random_state=RANDOM_STATE,
                    n_neighbors=30, min_dist=0.15, metric="euclidean",
                    target_metric="categorical")
    X_umap = umap_sup.fit_transform(X_scaled, y=y)
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(class_names):
        mask = (y == idx)
        plt.scatter(X_umap[mask, 0], X_umap[mask, 1],
                    marker=markers[idx % len(markers)],
                    label=name, s=60, linewidths=0.5)
    plt.xlabel("UMAP 1", fontsize=16, fontweight='bold')
    plt.ylabel("UMAP 2", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.legend(title="Class", title_fontsize=12, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTDIR / "umap_supervised.png", dpi=300)
    plt.close()

    # 13) Save preds
    pred_df = pd.DataFrame({"y_true": [class_names[i] for i in y_test],
                            "y_pred": [class_names[i] for i in y_pred]})
    pred_df.to_csv(OUTDIR / "test_predictions.csv", index=False)

    # Summary
    print("\n=== Saved Outputs ===")
    print(f"- Features (final CSV): {FINAL_CSV.resolve() if FINAL_CSV.exists() else 'generated this run'}")
    print(f"- Features (scaled XLSX): {FEATURES_XLSX.resolve()}")
    print(f"- Confusion Matrix: {str((OUTDIR / 'confusion_matrix.png').resolve())}")
    print(f"- ROC Curves:       {str((OUTDIR / 'roc_curves.png').resolve())}")
    print(f"- t-SNE Plot:       {str((OUTDIR / 'tsne.png').resolve())}")
    print(f"- UMAP Supervised:  {str((OUTDIR / 'umap_supervised.png').resolve())}")
    print(f"- Report (txt):     {str((OUTDIR / 'classification_report.txt').resolve())}")
    print(f"- ROC AUC (txt):    {str((OUTDIR / 'roc_auc_report.txt').resolve())}")
    print(f"- Confusion CSV:    {str((OUTDIR / 'confusion_matrix.csv').resolve())}")
    print(f"- Predictions CSV:  {str((OUTDIR / 'test_predictions.csv').resolve())}")

if __name__ == "__main__":
    main()
