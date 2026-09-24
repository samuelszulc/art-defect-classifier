"""Feature extraction from ART impulse responses.

Features are physically interpretable, which matters in quality control:
an inspector should be able to see *why* a part was rejected.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .simulate import FS, PartModel

BAND = 0.08  # search +/-8% around each nominal mode frequency


def spectrum(x, fs=FS):
    """Magnitude spectrum with a Hann window, zero-padded for finer frequency resolution."""
    n = 1 << int(np.ceil(np.log2(len(x) * 4)))
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return freqs, mag


def _half_power_bandwidth(freqs, mag, idx):
    """-3 dB bandwidth of the peak at idx, used to estimate damping (zeta ~ bw / 2f)."""
    level = mag[idx] / np.sqrt(2)
    lo = idx
    while lo > 0 and mag[lo] > level:
        lo -= 1
    hi = idx
    while hi < len(mag) - 1 and mag[hi] > level:
        hi += 1
    return freqs[hi] - freqs[lo]


def extract_features(x, fs=FS, model=None):
    model = model or PartModel()
    freqs, mag = spectrum(x, fs)
    mag_norm = mag / mag.max()
    feats = {}
    peak_freqs = []

    for i, f0 in enumerate(model.freqs, start=1):
        band = (freqs > f0 * (1 - BAND)) & (freqs < f0 * (1 + BAND))
        idx_band = np.where(band)[0]
        local = idx_band[np.argmax(mag[band])]
        fp = freqs[local]
        peak_freqs.append(fp)

        bw = _half_power_bandwidth(freqs, mag, local)
        # Count distinct, prominent peaks (mode splitting). Peaks must reach 25% of the
        # local maximum and be at least 15 Hz apart, which filters out noise ripples.
        min_dist = max(1, int(15 / (freqs[1] - freqs[0])))
        peaks, _ = find_peaks(mag[band], height=mag[local] * 0.25,
                              prominence=mag[local] * 0.25, distance=min_dist)

        feats[f"f{i}_hz"] = fp
        feats[f"f{i}_amp_rel"] = mag_norm[local]
        feats[f"f{i}_damping"] = bw / (2 * fp)
        feats[f"f{i}_n_peaks"] = len(peaks)

    # Frequency ratios cancel out the common geometry/material scatter of good parts
    for i in range(2, len(peak_freqs) + 1):
        feats[f"ratio_f{i}_f1"] = peak_freqs[i - 1] / peak_freqs[0]

    # Global descriptors
    power = mag ** 2
    feats["spectral_centroid_hz"] = float(np.sum(freqs * power) / np.sum(power))
    env = np.abs(x)
    n = len(env) // 10
    first, last = np.mean(env[:n] ** 2), np.mean(env[-n:] ** 2)
    feats["decay_db"] = 10 * np.log10(first / max(last, 1e-12))
    return feats


def extract_batch(signals, fs=FS, model=None):
    return pd.DataFrame([extract_features(x, fs, model) for x in signals])
