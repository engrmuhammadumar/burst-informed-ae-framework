import numpy as np
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

def td_features(x):
    x = x.astype(float)
    N = len(x)
    rms = np.sqrt(np.mean(x**2))
    mav = np.mean(np.abs(x))
    cf = np.max(np.abs(x)) / (rms + 1e-12)
    imp = np.max(np.abs(x)) / (mav + 1e-12)
    zcr = ((x[:-1] * x[1:]) < 0).mean()
    return {
        "RMS": rms, "MAV": mav, "STD": x.std(), "MEAN": x.mean(),
        "SKEW": skew(x, bias=False), "KURT": kurtosis(x, fisher=False, bias=False),
        "CREST": cf, "IMPULSE": imp, "ZCR": zcr
    }

def spectral_features(x, fs):
    X = np.abs(rfft(x))
    f = rfftfreq(len(x), d=1/fs)
    psd = X**2
    psd_sum = psd.sum() + 1e-12
    centroid = (f * psd).sum() / psd_sum
    dom_idx = np.argmax(psd)
    dom_f = f[dom_idx]
    bw = np.sqrt(((f - centroid)**2 * psd).sum() / psd_sum)
    p_norm = psd / psd_sum
    spec_entropy = -np.sum(p_norm * (np.log(p_norm + 1e-12)))
    # crude THD proxy: energy off the dominant bin
    thd = np.sqrt((psd_sum - psd[dom_idx]) / (psd[dom_idx] + 1e-12))
    return {
        "SPEC_CENT": centroid, "DOM_FREQ": dom_f, "BANDWIDTH": bw,
        "SPEC_ENT": spec_entropy, "THD": thd, "ENERGY": psd_sum
    }

def tfd_features(x, wavelet="morl", scales=32):
    # CWT scalogram statistics
    widths = np.linspace(1, scales, scales)
    cwtm, _ = pywt.cwt(x, widths, wavelet)
    P = np.abs(cwtm)**2
    E = P.sum()
    mean = P.mean()
    var = P.var()
    # simple contrast: (max-min)/mean
    contrast = (P.max() - P.min()) / (mean + 1e-12)
    entropy = -np.sum((P/E) * np.log((P/E) + 1e-12))
    return {"TF_ENERGY": E, "TF_VAR": var, "TF_CONTRAST": contrast, "TF_ENTROPY": entropy}

def hos_features(x):
    # simple higher-order cumulant proxies
    x = (x - x.mean()) / (x.std() + 1e-12)
    m3 = np.mean(x**3)
    m5 = np.mean(x**5)
    return {"MOMENT3": m3, "MOMENT5": m5}

def burst_features(x, segs, fs):
    if not segs:
        return {"BURST_COUNT": 0, "BURST_RATE": 0, "MEAN_IBI": 0, "MEAN_RISET": 0, "BURST_ENERGY": 0}
    count = len(segs)
    centers = np.array([(s+e)//2 for s, e in segs], dtype=float) / fs
    ibi = np.diff(centers) if len(centers) > 1 else np.array([0.0])
    energies = [np.sum(x[s:e]**2) for s, e in segs]
    # rise time: time from 10% to 90% of peak in each burst
    risets = []
    for s, e in segs:
        y = np.abs(x[s:e]); peak = y.max(); th1, th2 = 0.1*peak, 0.9*peak
        i1 = np.argmax(y >= th1); i2 = np.argmax(y >= th2)
        risets.append(max(0, i2 - i1)/fs)
    return {
        "BURST_COUNT": count,
        "BURST_RATE": count / (len(x)/fs),
        "MEAN_IBI": float(np.mean(ibi)) if ibi.size else 0.0,
        "MEAN_RISET": float(np.mean(risets)) if risets else 0.0,
        "BURST_ENERGY": float(np.mean(energies))
    }

def extract_all_domains(x, fs, segs, tfd_cfg):
    feats = {}
    feats.update(td_features(x))
    feats.update(spectral_features(x, fs))
    feats.update(tfd_features(x, wavelet=tfd_cfg["cwt_wavelet"], scales=tfd_cfg["scales"]))
    feats.update(hos_features(x))
    feats.update(burst_features(x, segs, fs))
    return feats
