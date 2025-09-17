import numpy as np
import pywt
from scipy.signal import convolve

def _mad(x):
    return np.median(np.abs(x - np.median(x))) + 1e-12

def wavelet_denoise(x, wavelet="db4", level=None, alpha=0.8):
    # DWT
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    level = level or max_level - 1 if max_level > 1 else 1
    coeffs = pywt.wavedec(x, wavelet, level=level)
    # adaptive thresholds using MAD per level
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        sigma = _mad(c) / 0.6745
        thr = sigma * np.sqrt(2.0 * np.log(len(c)))
        # hybrid soft-hard
        soft = np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)
        hard = c * (np.abs(c) >= thr)
        coeffs[i] = alpha * soft + (1 - alpha) * hard
    return pywt.waverec(coeffs, wavelet)

def short_time_energy(x, win):
    w = np.ones(win, dtype=float) / win
    e = convolve(x**2, w, mode="same")
    return e

def burst_segments(x, fs, win_ms=1.0, alpha=3.0, pad_ms=0.2, max_segments=10):
    win = max(1, int(fs * win_ms * 1e-3))
    pad = max(1, int(fs * pad_ms * 1e-3))
    E = short_time_energy((x - x.mean()) / (x.std() + 1e-12), win)
    baseline = np.median(E) + 1e-12
    idx = np.where(E / baseline >= alpha)[0]
    if idx.size == 0:
        return [(0, len(x))]
    # merge contiguous indices into segments
    segs = []
    start = idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            segs.append((max(0, start - pad), min(len(x), idx[i-1] + pad)))
            start = idx[i]
    segs.append((max(0, start - pad), min(len(x), idx[-1] + pad)))
    # clip to max_segments by energy
    seg
