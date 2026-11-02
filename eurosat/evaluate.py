"""Evaluate a trained run on the held-out test split.

Writes into the run folder:
  metrics.json            accuracy, macro-F1, per-class precision/recall/F1
  confusion_matrix.png    row-normalised (recall) confusion matrix
  confused_pairs.csv      class pairs ranked by symmetric confusion rate
  predictions.csv         one row per test image: true, pred, confidence, all probs

--tta averages predictions over the 8 flips/rotations of each patch. Because
satellite patches have no orientation, this is a free accuracy boost and a
nice small result to report (with vs without TTA).

Usage:
  python evaluate.py --run outputs/rgb_mobilenetv2 [--tta]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config
import data
from model import load


def dihedral_views(X):
    views = []
    for k in range(4):
        r = np.rot90(X, k, axes=(1, 2))
        views.append(r)
        views.append(r[:, :, ::-1])
    return views


def predict(model, X, tta=False, batch=256):
    if not tta:
        return model.predict(X.astype(np.float32), batch_size=batch, verbose=0)
    probs = [model.predict(np.ascontiguousarray(v).astype(np.float32), batch_size=batch, verbose=0)
             for v in dihedral_views(X)]
    return np.mean(probs, axis=0)


def confused_pairs(cm_norm):
    rows = []
    n = len(config.CLASS_NAMES)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = config.CLASS_NAMES[i], config.CLASS_NAMES[j]
            rows.append({"class_a": a, "class_b": b,
                         "a_as_b": cm_norm[i, j], "b_as_a": cm_norm[j, i],
                         "symmetric_rate": (cm_norm[i, j] + cm_norm[j, i]) / 2})
    return pd.DataFrame(rows).sort_values("symmetric_rate", ascending=False).reset_index(drop=True)


def plot_cm(cm_norm, path, title):
    labels = [config.PRETTY_NAMES[c] for c in config.CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="Run folder containing best_model.keras and run_config.json")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    with open(os.path.join(args.run, "run_config.json")) as f:
        run_cfg = json.load(f)
    arrays, _, _ = data.prepare(run_cfg["composite"], args.data_root)
    X, y, ids = arrays[args.split]
    model = load(os.path.join(args.run, "best_model.keras"))

    probs = predict(model, X, tta=args.tta)
    pred = probs.argmax(1)
    suffix = "_tta" if args.tta else ""

    labels = list(range(len(config.CLASS_NAMES)))
    cm = confusion_matrix(y, pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    report = classification_report(y, pred, labels=labels, target_names=config.CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "split": args.split, "tta": args.tta, "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "per_class": {c: report[c] for c in config.CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(args.run, f"metrics{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    title = f"{run_cfg['composite']} / {run_cfg['backbone']}  -  acc {metrics['accuracy'] * 100:.2f}%"
    plot_cm(cm_norm, os.path.join(args.run, f"confusion_matrix{suffix}.png"), title)

    pairs = confused_pairs(cm_norm)
    pairs.to_csv(os.path.join(args.run, f"confused_pairs{suffix}.csv"), index=False)

    df = pd.DataFrame({"id": ids, "true": [config.CLASS_NAMES[i] for i in y],
                       "pred": [config.CLASS_NAMES[i] for i in pred],
                       "confidence": probs.max(1), "correct": pred == y})
    for k, c in enumerate(config.CLASS_NAMES):
        df[f"p_{c}"] = probs[:, k]
    df.to_csv(os.path.join(args.run, f"predictions{suffix}.csv"), index=False)

    print(f"{args.split} accuracy: {metrics['accuracy'] * 100:.2f}%   macro-F1: {metrics['macro_f1'] * 100:.2f}%")
    print("\nPer-class F1:")
    for c in config.CLASS_NAMES:
        print(f"  {c:22s} {report[c]['f1-score'] * 100:6.2f}%")
    print("\nMost confused pairs (symmetric rate):")
    for _, r in pairs.head(5).iterrows():
        print(f"  {r.class_a:>20s} <-> {r.class_b:<20s} {r.symmetric_rate * 100:5.2f}%  "
              f"({r.a_as_b * 100:.2f}% / {r.b_as_a * 100:.2f}%)")
    print(f"\nNext: python confusability.py --run {args.run}")


if __name__ == "__main__":
    main()
