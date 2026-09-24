"""Classify a single part from its tap recording.

Usage:
    python predict.py recording.wav      # mono WAV file (resampled to 51.2 kHz if needed)
    python predict.py --demo             # simulate one OK and one cracked part
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from art.features import extract_features
from art.simulate import FS, simulate_part

MODEL_PATH = "models/art_rf.joblib"


def load_wav(path):
    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != FS:
        x = resample(x, int(len(x) * FS / fs))
    # Start the analysis at the tap: first sample above 10% of the peak
    start = int(np.argmax(np.abs(x) > 0.1 * np.abs(x).max()))
    return x[start:start + int(0.2 * FS)]


def classify(bundle, x):
    feats = pd.DataFrame([extract_features(x)])[bundle["features"]]
    proba = bundle["model"].predict_proba(feats)[0, 1]
    return ("DEFECT" if proba >= 0.5 else "OK"), proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", nargs="?")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    try:
        bundle = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit("Model not found. Run `python train.py` first.")

    if args.demo:
        rng = np.random.default_rng(123)
        for defective in (False, True):
            x, sev = simulate_part(defective, rng, severity=0.6 if defective else None)
            verdict, p = classify(bundle, x)
            truth = f"cracked (severity {sev:.1f})" if defective else "good"
            print(f"Simulated {truth:24s} -> {verdict:6s} (defect probability {p:.2f})")
    elif args.wav:
        verdict, p = classify(bundle, load_wav(args.wav))
        print(f"{args.wav}: {verdict} (defect probability {p:.2f})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
