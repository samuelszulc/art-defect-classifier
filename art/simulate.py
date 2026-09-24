"""Synthetic impulse responses for acoustic resonance testing (ART).

A tapped part rings at its natural frequencies. Each mode is modelled as an
exponentially damped sinusoid, so the recorded signal is

    x(t) = sum_i A_i * exp(-2*pi*f_i*zeta_i*t) * sin(2*pi*f_i*t + phi_i) + noise

A crack reduces local stiffness, which lowers some natural frequencies
(each mode by a different amount, depending on where the crack sits relative
to the mode shape), increases damping and can split a mode into two close peaks.

Good parts are not identical either: tolerances in geometry and material shift
all frequencies together by a similar factor. This overlap is what makes a
simple frequency threshold unreliable and motivates the ML approach.
"""
from dataclasses import dataclass, field

import numpy as np

FS = 51_200          # sampling rate [Hz]
DURATION = 0.2       # recording length [s]


@dataclass
class PartModel:
    """Nominal modal parameters of the tested part."""
    freqs: np.ndarray = field(default_factory=lambda: np.array([2150.0, 5480.0, 9320.0, 13800.0]))
    amps: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.2, 1.4, 1.6]))
    zetas: np.ndarray = field(default_factory=lambda: np.array([0.0020, 0.0018, 0.0016, 0.0015]))
    # Relative frequency drop per mode for a crack of severity 1.0
    crack_sensitivity: np.ndarray = field(default_factory=lambda: np.array([0.006, 0.020, 0.009, 0.028]))
    # Modes that split into two peaks when cracked
    split_modes: tuple = (1, 3)


def _ring(t, freqs, amps, zetas, rng):
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    x = np.zeros_like(t)
    for f, a, z, p in zip(freqs, amps, zetas, phases):
        x += a * np.exp(-2 * np.pi * f * z * t) * np.sin(2 * np.pi * f * t + p)
    return x


def simulate_part(defective, rng, model=None, severity=None, snr_db=None):
    """Return (signal, severity) for one tapped part."""
    model = model or PartModel()
    t = np.arange(int(FS * DURATION)) / FS

    # Manufacturing tolerance: one common factor for all modes + small per-mode scatter
    global_scale = 1 + rng.normal(0, 0.006)
    freqs = model.freqs * global_scale * (1 + rng.normal(0, 0.0015, len(model.freqs)))
    # Tap position and force change relative mode amplitudes
    amps = model.amps * rng.uniform(0.75, 1.25, len(model.amps)) * rng.uniform(0.5, 1.5)
    zetas = model.zetas * rng.uniform(0.75, 1.25) * rng.uniform(0.9, 1.1, len(model.zetas))

    extra_freqs, extra_amps, extra_zetas = [], [], []
    if defective:
        severity = rng.uniform(0.08, 1.0) if severity is None else severity
        freqs = freqs * (1 - severity * model.crack_sensitivity * rng.uniform(0.7, 1.3, len(freqs)))
        zetas = zetas * (1 + 1.0 * severity * rng.uniform(0.5, 1.5, len(zetas)))
        for m in model.split_modes:
            if rng.random() < 0.3 + 0.6 * severity:
                split = freqs[m] * 0.006 * severity
                extra_freqs.append(freqs[m] - split)
                extra_amps.append(amps[m] * rng.uniform(0.3, 0.7))
                extra_zetas.append(zetas[m])
    else:
        severity = 0.0

    x = _ring(t, np.concatenate([freqs, extra_freqs]), np.concatenate([amps, extra_amps]),
              np.concatenate([zetas, extra_zetas]), rng)

    snr_db = rng.uniform(40, 55) if snr_db is None else snr_db
    noise_power = np.mean(x ** 2) / (10 ** (snr_db / 10))
    x = x + rng.normal(0, np.sqrt(noise_power), len(x))
    return x.astype(np.float32), float(severity)


def make_dataset(n_ok=600, n_defect=200, seed=42, model=None):
    """Generate a labelled dataset. Label 0 = OK, 1 = defective (crack)."""
    rng = np.random.default_rng(seed)
    signals, labels, severities = [], [], []
    for label, n in ((0, n_ok), (1, n_defect)):
        for _ in range(n):
            x, s = simulate_part(bool(label), rng, model)
            signals.append(x)
            labels.append(label)
            severities.append(s)
    order = rng.permutation(len(labels))
    return (np.stack(signals)[order], np.array(labels)[order], np.array(severities)[order])
