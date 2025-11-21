# rigorous_end_to_end.py
# End-to-end, leakage-free, group-aware pipeline for AE diagnostic modeling
# 1) Build features from MAT -> Excel with GroupID
# 2) Group-aware nested CV (outer=5, inner=3) with pipelines
# 3) Summary tables, stats, and plots for rebuttal/manuscript

import os
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIG: Update these
# =========================
MAT_PATH = r"E:\1 Paper MCT\Cutting Tool Paper\Dataset\cutting tool data\mat files data\AE_ALL.mat"
OUT_DIR  = r"E:\2 Paper MCT\Multi-Domain Paper\Review 1 EFA\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(OUT_DIR, "full_ae_features_WITH_GROUPID.xlsx")

# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import h5py
import pywt

from scipy.signal import welch, find_peaks, hilbert
from scipy.stats import skew, kurtosis, entropy
from numpy.fft import fft2

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc, brier_score_loss)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon


# =========================
# 1) DATA LOADING & FEATURES
# =========================
def load_raw_signals_from_mat(mat_path, class_map, channel_index=0):
    """
    Loads raw AE signals from the MATLAB structure, returning:
    raw_signals: dict[class] -> list of 1D arrays (40 parent signals per class)
    """
    raw_signals = {}
    with h5py.File(mat_path, 'r') as mat_file:
        ae_all = mat_file['AE_ALL']
        for label, mat_key in class_map.items():
            cls_refs = ae_all[mat_key]     # MATLAB cell-like (1, 4); we need channel_index
            ref = cls_refs[channel_index][0]
            dataset = mat_file[ref]
            signal_array = np.array(dataset)  # shape: (samples, 40)
            signals = [signal_array[:, i] for i in range(signal_array.shape[1])]
            raw_signals[label] = signals
            print(f"{label}: Loaded {len(signals)} parent signals, each with shape {signals[0].shape}")
    return raw_signals


def segment_signal(signal, frame_size=10000, num_frames=10):
    segments = []
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        segments.append(signal[start:end])
    return segments


def denoise_wavelet(frame, wavelet='db4', level=3):
    coeffs = pywt.wavedec(frame, wavelet=wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(frame)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs_thresh, wavelet=wavelet)
    return denoised[:len(frame)]


def extract_td_fd_features(frame, fs=1e6):
    feats = []
    # Time domain
    mean_val = np.mean(frame)
    std_val = np.std(frame)
    var_val = np.var(frame)
    rms_val = np.sqrt(np.mean(frame**2))
    peak_val = np.max(np.abs(frame))
    ptp_val = np.ptp(frame)
    crest_factor = peak_val / rms_val if rms_val != 0 else 0
    mean_abs = np.mean(np.abs(frame))
    impulse_factor = peak_val / mean_abs if mean_abs != 0 else 0
    shape_factor = rms_val / mean_abs if mean_abs != 0 else 0
    margin_factor = peak_val / (np.mean(np.sqrt(np.abs(frame)))**2) if np.mean(np.sqrt(np.abs(frame))) != 0 else 0
    skewness = skew(frame)
    kurt = kurtosis(frame)
    energy = np.sum(frame**2)
    zcr = ((frame[:-1] * frame[1:]) < 0).sum()
    mav = mean_abs
    max_val = np.max(frame)
    min_val = np.min(frame)
    range_val = max_val - min_val
    duration = len(frame) / fs
    envelope_area = np.trapz(np.abs(frame))

    # Frequency domain (Welch)
    freqs, psd = welch(frame, fs=fs, nperseg=1024)
    psd = np.nan_to_num(psd)
    spectral_energy = np.sum(psd)
    denom = np.sum(psd) if np.sum(psd) != 0 else 1.0
    spectral_centroid = np.sum(freqs * psd) / denom
    spectral_entropy = entropy(psd + 1e-12)
    spectral_kurtosis = kurtosis(psd)
    spectral_skewness = skew(psd)
    spectral_flatness = np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12)
    dominant_freq = freqs[np.argmax(psd)]
    cumsum = np.cumsum(psd)
    median_freq = freqs[np.searchsorted(cumsum, cumsum[-1]/2)]
    mean_freq = spectral_centroid
    freq_variance = np.var(psd)
    freq_rms = np.sqrt(np.mean(psd**2))
    freq_spread = np.sqrt(np.sum(((freqs - mean_freq)**2) * psd) / denom)
    # THD (rough)
    peaks, _ = find_peaks(psd, height=np.max(psd)*0.1)
    thd = (np.sum(psd[peaks[1:]]) / (psd[peaks[0]] + 1e-12)) if len(peaks) > 1 else 0

    feats.extend([
        mean_val, rms_val, std_val, var_val,
        skewness, kurt, peak_val, ptp_val,
        crest_factor, impulse_factor, shape_factor, margin_factor,
        energy, max_val, min_val, range_val,
        zcr, duration, mav, envelope_area,
        mean_freq, median_freq, dominant_freq, spectral_centroid,
        spectral_entropy, spectral_kurtosis, spectral_skewness,
        spectral_flatness, spectral_energy, freq_variance,
        freq_rms, freq_spread, thd
    ])
    return feats


def extract_tfd_features_cwt(frame, wavelet='cmor1.5-1.0', fs=1e6, num_scales=64):
    scales = np.arange(1, num_scales + 1)
    coef, _ = pywt.cwt(frame, scales=scales, wavelet=wavelet, sampling_period=1/fs)
    scalogram = np.abs(coef)
    flat = scalogram.flatten()
    tfd_energy = np.sum(scalogram**2)
    tfd_entropy = entropy(flat + 1e-12)
    tfd_mean = np.mean(scalogram)
    tfd_std = np.std(scalogram)
    tfd_max = np.max(scalogram)
    tfd_kurt = kurtosis(flat)
    tfd_skew = skew(flat)
    # centroid over scale index:
    tfd_centroid = (np.sum(scalogram * np.arange(scalogram.shape[0])[:, None]) /
                    (np.sum(scalogram) + 1e-12))
    top_k = np.percentile(flat, 95)
    ridge_energy = np.sum(flat[flat >= top_k]) / (np.sum(flat) + 1e-12)
    return [tfd_energy, tfd_entropy, tfd_mean, tfd_std, tfd_max, tfd_kurt, tfd_skew, tfd_centroid, ridge_energy]


def extract_hos_features(frame):
    centered = frame - np.mean(frame)
    m3 = np.mean(centered**3)
    m4 = np.mean(centered**4)
    m5 = np.mean(centered**5)
    c3 = m3
    c4 = m4 - 3*(np.var(frame)**2)
    g_index = c4 / (m4 + 1e-12)
    # Bispectrum crude approx via analytic signal outer product
    analytic = hilbert(frame)
    N = len(frame) // 2
    X = fft2(np.outer(analytic[:N], analytic[:N]))
    bispec = np.abs(X)
    bicoherence = bispec / (np.outer(np.abs(analytic[:N])**2, np.abs(analytic[:N])**2) + 1e-12)
    bispec_mean = np.mean(bispec)
    bicoherence_mean = np.mean(bicoherence)
    nonlinearity_index = c3**2 + c4**2
    return [m3, m4, m5, c3, c4, g_index, bispec_mean, bicoherence_mean, nonlinearity_index]


def extract_burst_features(frame, fs=1e6, threshold_ratio=0.25):
    thr = threshold_ratio * np.max(np.abs(frame))
    idx = np.where(np.abs(frame) > thr)[0]
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
    num_bursts = len(durations)
    # slice each burst segment safely:
    burst_energy = []
    burst_peak = []
    rms_burst = []
    for s, d in zip(starts, durations):
        end = min(len(frame), s + int(d*fs))
        seg = frame[s:end]
        if len(seg) == 0:
            continue
        burst_energy.append(np.sum(seg**2))
        burst_peak.append(np.max(np.abs(seg)))
        rms_burst.append(np.sqrt(np.mean(seg**2)))
    ibi = np.diff(starts)/fs if len(starts) > 1 else np.array([])
    total_event_duration = np.sum(durations)
    cumulative_counts = int(np.sum([int(d*fs) for d in durations]))
    return [
        num_bursts,
        float(np.mean(durations)) if durations else 0.0,
        float(np.mean(ibi)) if ibi.size else 0.0,
        float(np.mean(burst_energy)) if burst_energy else 0.0,
        float(np.mean(burst_peak)) if burst_peak else 0.0,
        float(np.mean(rms_burst)) if rms_burst else 0.0,
        cumulative_counts,
        total_event_duration,
        float(np.max(burst_peak)) if burst_peak else 0.0,
        float(np.min(burst_peak)) if burst_peak else 0.0
    ]


def build_features_with_groupid(mat_path, out_excel_path,
                                frame_size=10000, num_frames=10, fs=1e6):
    # Map MATLAB keys -> desired labels (same as your earlier code)
    class_map = {
        'BF': 'BF',
        'GF': 'GF',
        'TF': 'BFI',  # Use BFI but label as TF
        'N':  'NI'    # Use NI but label as N
    }
    raw = load_raw_signals_from_mat(mat_path, class_map, channel_index=0)

    # Segment per parent; denoise per frame; NO burst selection here (avoid leakage)
    # We will create one row per frame with GroupID "<Class>_<parent_idx>"
    td_fd_rows, tfd_rows, hos_rows, burst_rows = [], [], [], []
    labels, group_ids = [], []

    print("\nSegmenting and denoising...")
    for cls, signals in raw.items():         # 40 parent signals per class
        for sig_idx, sig in enumerate(signals):
            frames = segment_signal(sig, frame_size=frame_size, num_frames=num_frames)
            for frame in frames:
                d = denoise_wavelet(frame)
                td = extract_td_fd_features(d, fs=fs)
                tfd = extract_tfd_features_cwt(d, fs=fs)
                hos = extract_hos_features(d)
                bf  = extract_burst_features(d, fs=fs)

                td_fd_rows.append(td)
                tfd_rows.append(tfd)
                hos_rows.append(hos)
                burst_rows.append(bf)
                labels.append(cls)
                group_ids.append(f"{cls}_{sig_idx}")

    # Names
    td_fd_names = [
        "Mean","RMS","STD","Variance","Skewness","Kurtosis","Peak","PeakToPeak",
        "CrestFactor","ImpulseFactor","ShapeFactor","MarginFactor","SignalEnergy",
        "MaxVal","MinVal","Range","ZCR","Duration","MAV","EnvelopeArea",
        "MeanFreq","MedianFreq","DominantFreq","SpectralCentroid","SpectralEntropy",
        "SpectralKurtosis","SpectralSkewness","SpectralFlatness","SpectralEnergy",
        "FreqVariance","FreqRMS","FreqSpread","THD"
    ]
    tfd_names = [
        "TFD_Energy","TFD_Entropy","TFD_Mean","TFD_STD","TFD_Max",
        "TFD_Kurtosis","TFD_Skewness","TFD_Centroid","TFD_RidgeEnergyRatio"
    ]
    hos_names = [
        "Moment3","Moment4","Moment5","Cumulant3","Cumulant4",
        "GaussianityIndex","BispectrumMean","BicoherenceMean","NonlinearityIndex"
    ]
    burst_names = [
        "NumBursts","AvgBurstDuration","AvgInterBurstInterval","AvgBurstEnergy",
        "AvgBurstPeak","AvgBurstRMS","CumulativeCounts","TotalEventDuration",
        "MaxBurstPeak","MinBurstPeak"
    ]

    td_fd_df = pd.DataFrame(td_fd_rows, columns=td_fd_names)
    tfd_df   = pd.DataFrame(tfd_rows,   columns=tfd_names)
    hos_df   = pd.DataFrame(hos_rows,   columns=hos_names)
    burst_df = pd.DataFrame(burst_rows, columns=burst_names)

    combined = pd.concat([td_fd_df, tfd_df, hos_df, burst_df], axis=1)
    combined["Label"]   = labels
    combined["GroupID"] = group_ids

    combined.to_excel(out_excel_path, index=False)
    print(f"\n✅ Features saved: {out_excel_path}")
    return combined


# =========================
# 2) GROUP-AWARE NESTED CV
# =========================
def infer_feature_families(columns):
    td_fd, tfd, hos, burst = [], [], [], []
    for c in columns:
        if c.startswith("TFD_"):
            tfd.append(c)
        elif c.startswith(("Moment", "Cumulant", "GaussianityIndex", "Bispectrum", "Bicoherence", "NonlinearityIndex")):
            hos.append(c)
        elif c in ("NumBursts","AvgBurstDuration","AvgInterBurstInterval","AvgBurstEnergy","AvgBurstPeak",
                   "AvgBurstRMS","CumulativeCounts","TotalEventDuration","MaxBurstPeak","MinBurstPeak"):
            burst.append(c)
        elif c not in ("Label","GroupID","EncodedLabel"):
            td_fd.append(c)
    families = {"TD_FD": td_fd, "TFD": tfd, "HOS": hos, "BURST": burst, "ALL": td_fd + tfd + hos + burst}
    return families


def mean_ci(a):
    mu = np.mean(a); sd = np.std(a, ddof=1)
    return mu, sd, (mu - 1.96*sd, mu + 1.96*sd)


def nested_eval_all_models(df, out_dir):
    y = df["Label"].values
    groups = df["GroupID"].values
    feature_cols = [c for c in df.columns if c not in ("Label","EncodedLabel","GroupID")]
    X = df[feature_cols].values
    classes = np.unique(y)
    n_classes = len(classes)

    # Models (pipelines)
    pipelines = {
        "RF":  Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        "LR":  Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, multi_class="auto", random_state=42))]),
    }

    param_grids = {
        "RF":  {"clf__n_estimators":[100,200],"clf__max_depth":[None,10,20],"clf__min_samples_split":[2,5]},
        "SVM": {"clf__C":[0.5,1,5,10],"clf__gamma":["scale",1e-3,1e-4]},
        "KNN": {"clf__n_neighbors":[3,5,7,9],"clf__weights":["uniform","distance"]},
        "LR":  {"clf__C":[0.1,1,10],"clf__penalty":["l2"],"clf__solver":["lbfgs"]},
    }

    def nested_eval(model_key):
        pipe, grid = pipelines[model_key], param_grids[model_key]
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)

        accs, f1s, cms = [], [], []
        probs_all, y_all = [], []

        print(f"\n== {model_key}: group-aware nested CV ==")
        for k,(tr,te) in enumerate(outer.split(X, y, groups), start=1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            gtr      = groups[tr]

            gs = GridSearchCV(pipe, grid, cv=list(inner.split(Xtr,ytr,gtr)), scoring="accuracy", n_jobs=-1)
            gs.fit(Xtr, ytr, groups=gtr)
            best = gs.best_estimator_

            yhat = best.predict(Xte)
            acc  = accuracy_score(yte, yhat)
            f1   = f1_score(yte, yhat, average="macro")
            accs.append(acc); f1s.append(f1)
            cms.append(confusion_matrix(yte, yhat, labels=classes))

            if hasattr(best, "predict_proba"):
                prob = best.predict_proba(Xte)
            else:
                dec = best.decision_function(Xte)
                dec = (dec - dec.min(axis=1, keepdims=True))
                prob = dec / (dec.sum(axis=1, keepdims=True) + 1e-12)

            probs_all.append(prob); y_all.append(yte)
            print(f"  Fold {k}: acc={acc:.4f}, F1={f1:.4f}, best={gs.best_params_}")

        return {
            "acc": np.array(accs), "f1": np.array(f1s),
            "probs": np.vstack(probs_all), "y": np.concatenate(y_all),
            "cms": cms
        }

    results = {m: nested_eval(m) for m in ["RF","SVM","KNN","LR"]}

    # Ensemble: tune base models per outer fold
    def ensemble_eval():
        print("\n== ENS: group-aware nested CV ==")
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        accs, f1s, cms = [], [], []
        probs_all, y_all = [], []
        for k,(tr,te) in enumerate(outer.split(X, y, groups), start=1):
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            gtr      = groups[tr]
            tuned = []
            for m in ["RF","SVM","KNN","LR"]:
                pipe = pipelines[m]; grid = param_grids[m]
                gs = GridSearchCV(pipe, grid, cv=list(inner.split(Xtr,ytr,gtr)), scoring="accuracy", n_jobs=-1)
                gs.fit(Xtr, ytr, groups=gtr)
                tuned.append((m, gs.best_estimator_))
            ens = VotingClassifier(tuned, voting="soft", n_jobs=-1)
            ens.fit(Xtr, ytr)
            yhat = ens.predict(Xte)
            acc  = accuracy_score(yte, yhat)
            f1   = f1_score(yte, yhat, average="macro")
            accs.append(acc); f1s.append(f1)
            cms.append(confusion_matrix(yte, yhat, labels=classes))
            prob = ens.predict_proba(Xte)
            probs_all.append(prob); y_all.append(yte)
            print(f"  Fold {k}: acc={acc:.4f}, F1={f1:.4f}")
        return {
            "acc": np.array(accs), "f1": np.array(f1s),
            "probs": np.vstack(probs_all), "y": np.concatenate(y_all),
            "cms": cms
        }

    results["ENS"] = ensemble_eval()

    # Summary table
    rows = []
    for k,v in results.items():
        mu_a, sd_a, (lo_a,hi_a) = mean_ci(v["acc"])
        mu_f, sd_f, (lo_f,hi_f) = mean_ci(v["f1"])
        rows.append([k, mu_a, sd_a, lo_a, hi_a, mu_f, sd_f, lo_f, hi_f])

    summary = pd.DataFrame(rows, columns=[
        "Model","Mean Acc","SD Acc","95% Acc Low","95% Acc High",
        "Mean Macro-F1","SD F1","95% F1 Low","95% F1 High"
    ])
    summary_path = os.path.join(out_dir, "group_nested_summary.xlsx")
    summary.to_excel(summary_path, index=False)
    print("✅ Saved:", summary_path)

    # Pick best by macro-F1
    means = {k:v["f1"].mean() for k,v in results.items()}
    best_key = max(means, key=means.get)
    best = results[best_key]
    print(f"\nBest (macro-F1): {best_key}  acc={best['acc'].mean():.4f}  f1={best['f1'].mean():.4f}")

    # Wilcoxon vs best
    print("\nWilcoxon vs BEST (macro-F1):")
    for k,v in results.items():
        if k==best_key: continue
        stat,p = wilcoxon(best["f1"], v["f1"])
        print(f"  {best_key} vs {k}: W={stat:.3f}, p={p:.4g}")

    # Plots: averaged confusion matrix (best)
    classes_sorted = np.unique(y)
    cm_mean = sum(best["cms"]) / len(best["cms"])
    fig,ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm_mean, cmap="Blues")
    ax.set_xticks(range(len(classes_sorted))); ax.set_yticks(range(len(classes_sorted)))
    ax.set_xticklabels(classes_sorted); ax.set_yticklabels(classes_sorted)
    plt.title(f"Averaged Confusion Matrix ({best_key})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(len(classes_sorted)):
        for j in range(len(classes_sorted)):
            ax.text(j, i, f"{cm_mean[i,j]:.1f}", ha="center", va="center")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"cm_mean.png"), dpi=300); plt.close()

    # ROC/PR (one-vs-rest) for best
    y_bin = label_binarize(best["y"], classes=classes_sorted)
    probs = best["probs"]

    # ROC
    plt.figure(figsize=(6,5))
    for i,c in enumerate(classes_sorted):
        fpr,tpr,_ = roc_curve(y_bin[:,i], probs[:,i])
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f"{c} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (one-vs-rest) — {best_key}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"roc_ovr.png"), dpi=300); plt.close()

    # PR
    plt.figure(figsize=(6,5))
    for i,c in enumerate(classes_sorted):
        prec,rec,_ = precision_recall_curve(y_bin[:,i], probs[:,i])
        pr_auc = auc(rec,prec)
        plt.plot(rec,prec,label=f"{c} (AUPR={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (one-vs-rest) — {best_key}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"pr_ovr.png"), dpi=300); plt.close()

    # Calibration (reliability) curve
    brier = brier_score_loss(y_bin.ravel(), probs.ravel())
    bins = np.linspace(0,1,11)
    binids = np.digitize(probs.ravel(), bins) - 1
    bin_true = [y_bin.ravel()[binids==b].mean() if np.any(binids==b) else np.nan for b in range(len(bins)-1)]
    bin_pred = [(bins[b]+bins[b+1])/2 for b in range(len(bins)-1)]
    plt.figure(figsize=(5,4))
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.plot(bin_pred, bin_true, marker='o')
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(f"Reliability Curve — {best_key}\nBrier={brier:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"calibration.png"), dpi=300); plt.close()

    # Quick bar chart for mean±sd accuracy across models
    labels = []
    means_acc = []
    errs = []
    for k,v in results.items():
        mu, sd, _ = mean_ci(v["acc"])
        labels.append(k); means_acc.append(mu); errs.append(sd)
    plt.figure(figsize=(7,4))
    plt.bar(labels, means_acc, yerr=errs, alpha=0.85, capsize=6)
    plt.ylabel("Accuracy (mean ± sd)"); plt.title("Nested CV — Models (ALL features)")
    plt.ylim(0,1.0); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"nestedcv_models_all.png"), dpi=300); plt.close()

    # LaTeX table
    latex_txt = summary.to_latex(index=False, float_format="%.4f")
    with open(os.path.join(out_dir, "nestedcv_summary_latex.txt"), "w") as f:
        f.write(latex_txt)
    print("✅ Saved:", os.path.join(out_dir, "nestedcv_summary_latex.txt"))

    return summary, results, best_key


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("="*80)
    print("START: Building features with GroupID (no leakage)")
    print("="*80)

    if not os.path.exists(EXCEL_PATH):
        df_features = build_features_with_groupid(MAT_PATH, EXCEL_PATH)
    else:
        print(f"Found existing features: {EXCEL_PATH}")
        df_features = pd.read_excel(EXCEL_PATH)

    print("\nDataset check:")
    print("  Samples:", len(df_features))
    print("  Features (total cols inc. Label/GroupID):", len(df_features.columns))
    print("  Classes:", sorted(df_features["Label"].unique()))
    print("  Groups:", len(df_features["GroupID"].unique()))

    print("\n" + "="*80)
    print("Nested CV (group-aware) with pipelines")
    print("="*80)

    summary, results, best_key = nested_eval_all_models(df_features, OUT_DIR)

    print("\nAll done. Key outputs:")
    print(" - Feature file:", EXCEL_PATH)
    print(" - Summary table:", os.path.join(OUT_DIR, "group_nested_summary.xlsx"))
    print(" - LaTeX table:", os.path.join(OUT_DIR, "nestedcv_summary_latex.txt"))
    print(" - Plots: cm_mean.png, roc_ovr.png, pr_ovr.png, calibration.png, nestedcv_models_all.png")
