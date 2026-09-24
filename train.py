"""Train and evaluate a Random Forest classifier for ART defect detection.

Usage:
    python train.py                 # default dataset (600 OK, 200 defective parts)
    python train.py --n-ok 1000 --n-defect 300 --seed 7
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from art.features import extract_batch, spectrum
from art.simulate import make_dataset

OUT_DOCS = "docs"
OUT_MODEL = "models"
LABELS = ["OK", "Defect"]


def threshold_baseline(f_train, y_train, f_test):
    """Classic ART rule: reject a part if its 2nd natural frequency is below a threshold.

    The threshold is tuned on the training set to maximise balanced accuracy.
    """
    candidates = np.quantile(f_train, np.linspace(0.01, 0.99, 200))
    scores = [balanced_accuracy_score(y_train, (f_train < c).astype(int)) for c in candidates]
    best = candidates[int(np.argmax(scores))]
    return (f_test < best).astype(int), best


def plot_example_spectra(signals, labels, severity, path):
    """One good part vs. one clearly cracked part: full spectrum and zoom on two modes."""
    ok = int(np.where(labels == 0)[0][0])
    bad = int(np.where((labels == 1) & (severity > 0.7))[0][0])
    spectra = {}
    for key, i in (("OK", ok), ("Defect", bad)):
        f, m = spectrum(signals[i])
        spectra[key] = (f / 1000, 20 * np.log10(m / m.max() + 1e-9))
    colors = {"OK": "tab:green", "Defect": "tab:red"}

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 1, 1)
    for key, (f, db) in spectra.items():
        ax.plot(f, db, color=colors[key], lw=0.8, alpha=0.85, label=key)
    ax.set(xlim=(1, 16), ylim=(-60, 3), xlabel="Frequency [kHz]", ylabel="Magnitude [dB]",
           title="Resonance spectrum: good part vs. cracked part")
    ax.legend()
    ax.grid(alpha=0.3)
    for k, (lo, hi, name) in enumerate(((5.25, 5.6, "Mode 2"), (13.05, 13.85, "Mode 4"))):
        ax = fig.add_subplot(2, 2, 3 + k)
        for key, (f, db) in spectra.items():
            ax.plot(f, db, color=colors[key], lw=1.2, label=key)
        ax.set(xlim=(lo, hi), ylim=(-60, 3), xlabel="Frequency [kHz]",
               title=f"{name}: frequency drop and peak splitting")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_importances(model, names, path, top=12):
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(names)[order], imp[order], color="tab:blue")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Most important features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-ok", type=int, default=600)
    parser.add_argument("--n-defect", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUT_DOCS, exist_ok=True)
    os.makedirs(OUT_MODEL, exist_ok=True)

    print(f"Simulating {args.n_ok} OK and {args.n_defect} defective parts...")
    signals, y, severity = make_dataset(args.n_ok, args.n_defect, args.seed)
    X = extract_batch(signals)
    print(f"Extracted {X.shape[1]} features per part.")

    X_tr, X_te, y_tr, y_te, sev_tr, sev_te = train_test_split(
        X, y, severity, test_size=0.25, stratify=y, random_state=args.seed)

    # Baseline: single-frequency threshold
    base_pred, base_thr = threshold_baseline(X_tr["f2_hz"].values, y_tr, X_te["f2_hz"].values)

    # Random Forest; class_weight balances the rarer defective class
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight="balanced", random_state=args.seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="balanced_accuracy")
    model.fit(X_tr, y_tr)
    rf_pred = model.predict(X_te)

    print("\n=== Baseline: threshold on 2nd natural frequency "
          f"(f2 < {base_thr:.0f} Hz -> defect) ===")
    print(classification_report(y_te, base_pred, target_names=LABELS, digits=3))
    print("=== Random Forest ===")
    print(f"5-fold CV balanced accuracy (train set): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_te, rf_pred, target_names=LABELS, digits=3))

    # Which defects are still missed? Usually the smallest cracks.
    missed = (y_te == 1) & (rf_pred == 0)
    if missed.any():
        print(f"Missed defects: {missed.sum()}, mean crack severity {sev_te[missed].mean():.2f} "
              f"(all defects: {sev_te[y_te == 1].mean():.2f})")

    metrics = {
        "dataset": {"n_ok": args.n_ok, "n_defect": args.n_defect, "seed": args.seed},
        "baseline_threshold_hz": round(float(base_thr), 1),
        "baseline": {
            "balanced_accuracy": round(balanced_accuracy_score(y_te, base_pred), 3),
            "defect_recall": round(recall_score(y_te, base_pred), 3),
            "false_reject_rate": round(float(((base_pred == 1) & (y_te == 0)).sum() / (y_te == 0).sum()), 3),
        },
        "random_forest": {
            "cv_balanced_accuracy": round(float(cv_scores.mean()), 3),
            "balanced_accuracy": round(balanced_accuracy_score(y_te, rf_pred), 3),
            "defect_recall": round(recall_score(y_te, rf_pred), 3),
            "false_reject_rate": round(float(((rf_pred == 1) & (y_te == 0)).sum() / (y_te == 0).sum()), 3),
        },
    }
    with open(os.path.join(OUT_DOCS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_example_spectra(signals, y, severity, os.path.join(OUT_DOCS, "spectra.png"))
    plot_importances(model, list(X.columns), os.path.join(OUT_DOCS, "feature_importance.png"))
    plot_confusion(y_te, base_pred, os.path.join(OUT_DOCS, "confusion_baseline.png"), "Threshold baseline")
    plot_confusion(y_te, rf_pred, os.path.join(OUT_DOCS, "confusion_rf.png"), "Random Forest")

    joblib.dump({"model": model, "features": list(X.columns)}, os.path.join(OUT_MODEL, "art_rf.joblib"))
    print(f"\nSaved model to {OUT_MODEL}/art_rf.joblib, plots and metrics to {OUT_DOCS}/")


if __name__ == "__main__":
    main()
