"""Confusability study: WHY does the model mix up specific class pairs?

For each of the top-k confused pairs (from evaluate.py) -- or a pair you name --
this produces evidence from four separate angles, so the report can argue
from more than just a few Grad-CAM pictures:

  1. Grad-CAM grids   correct A | A predicted as B | B predicted as A | correct B
  2. Attention focus  share of the image Grad-CAM marks as important (>0.5),
                      compared between correct and wrong predictions. "Errors
                      come with diffuse attention" is a testable claim.
  3. Input statistics how much the two classes overlap in raw colour and in
                      texture (edge density), measured with the Bhattacharyya
                      coefficient on the TRAINING images (1 = identical
                      distributions, 0 = fully separable). This separates
                      "the classes really look alike" from "the model is weak".
  4. Feature space    distance between the two class centroids in the model's
                      penultimate (GAP) embedding, plus a 2-D PCA scatter.

Outputs go to <run>/confusability/<A>__<B>/ plus a summary.csv across pairs.

Usage:
  python confusability.py --run outputs/rgb_mobilenetv2            # top 3 pairs
  python confusability.py --run outputs/rgb_mobilenetv2 --pair Highway River
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import data
from model import gradcam, last_conv_layer_name, load, make_gradcam_model, overlay

import keras


# ---------------------------------------------------------------- input statistics
def bhattacharyya(a, b, bins=32, rng=None):
    lo = min(a.min(), b.min()) if rng is None else rng[0]
    hi = max(a.max(), b.max()) if rng is None else rng[1]
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi))
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return float(np.sum(np.sqrt(ha * hb)))


def image_descriptors(X):
    """Per-image mean of each channel and mean gradient magnitude (texture)."""
    Xf = X.astype(np.float32)
    means = Xf.mean(axis=(1, 2))  # [N, 3]
    gray = Xf.mean(axis=3)
    gx = np.abs(np.diff(gray, axis=2)).mean(axis=(1, 2))
    gy = np.abs(np.diff(gray, axis=1)).mean(axis=(1, 2))
    return means, gx + gy


# ---------------------------------------------------------------- figures
def gradcam_grid(grad_model, groups, path, title, n_per_row=6):
    fig, axes = plt.subplots(len(groups), n_per_row, figsize=(2.2 * n_per_row, 2.5 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, (label, X, _) in enumerate(groups):
        if len(X):
            cams, probs = gradcam(grad_model, X[:n_per_row])
        else:
            axes[r, n_per_row // 2].text(0.5, 0.5, "no examples in test set", ha="center", va="center",
                                         transform=axes[r, n_per_row // 2].transAxes, color="gray")
        for c in range(n_per_row):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(X):
                ax.imshow(overlay(X[c], cams[c]))
                k = int(probs[c].argmax())
                ax.set_title(f"{config.PRETTY_NAMES[config.CLASS_NAMES[k]]} {probs[c, k] * 100:.0f}%", fontsize=8)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def raw_grid(groups, path, title, n_per_row=6):
    """Same layout without heatmaps, so readers can judge the images themselves."""
    fig, axes = plt.subplots(len(groups), n_per_row, figsize=(2.0 * n_per_row, 2.2 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, (label, X, _) in enumerate(groups):
        for c in range(n_per_row):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(X):
                ax.imshow(X[c], interpolation="nearest")
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, fontweight="bold")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def attention_focus(grad_model, X, batch=64):
    if len(X) == 0:
        return np.array([])
    out = []
    for i in range(0, len(X), batch):
        cams, _ = gradcam(grad_model, X[i:i + batch])
        out.append((cams > 0.5).mean(axis=(1, 2)))
    return np.concatenate(out)


# ---------------------------------------------------------------- per pair
def analyse_pair(a, b, model, grad_model, embed_model, arrays, preds, out_dir, n_show=6):
    ia, ib = config.CLASS_NAMES.index(a), config.CLASS_NAMES.index(b)
    Xte, yte, ids = arrays["test"]
    pred = preds.set_index("id").loc[ids, "pred"].map(config.CLASS_NAMES.index).to_numpy()

    sel = {
        "correct_a": (yte == ia) & (pred == ia),
        "a_as_b": (yte == ia) & (pred == ib),
        "b_as_a": (yte == ib) & (pred == ia),
        "correct_b": (yte == ib) & (pred == ib),
    }
    rng = np.random.default_rng(config.SEED)

    def sample(mask, n):
        idx = np.where(mask)[0]
        return Xte[rng.permutation(idx)[:n]] if len(idx) else Xte[:0]

    A, B = config.PRETTY_NAMES[a], config.PRETTY_NAMES[b]
    groups = [(f"{A} ✓", sample(sel["correct_a"], n_show), None),
              (f"{A} → {B}", sample(sel["a_as_b"], n_show), None),
              (f"{B} → {A}", sample(sel["b_as_a"], n_show), None),
              (f"{B} ✓", sample(sel["correct_b"], n_show), None)]
    os.makedirs(out_dir, exist_ok=True)
    gradcam_grid(grad_model, groups, os.path.join(out_dir, "gradcam_grid.png"),
                 f"Grad-CAM: {A} vs {B} (title = predicted class, confidence)")
    raw_grid(groups, os.path.join(out_dir, "raw_grid.png"), f"Test images: {A} vs {B}")

    # 2. attention focus: correct vs wrong, within this pair
    correct_X = Xte[sel["correct_a"] | sel["correct_b"]]
    wrong_X = Xte[sel["a_as_b"] | sel["b_as_a"]]
    focus_c = attention_focus(grad_model, correct_X[:300])
    focus_w = attention_focus(grad_model, wrong_X)

    # 3. input statistics on training images
    Xtr, ytr, _ = arrays["train"]
    ma, ta = image_descriptors(Xtr[ytr == ia])
    mb, tb = image_descriptors(Xtr[ytr == ib])
    channel_names = ("ch1", "ch2", "ch3")
    colour_overlap = {channel_names[k]: bhattacharyya(ma[:, k], mb[:, k], rng=(0, 255)) for k in range(3)}
    texture_overlap = bhattacharyya(ta, tb)

    # 4. feature space
    ea = embed_model.predict(Xtr[ytr == ia][:500].astype(np.float32), verbose=0)
    eb = embed_model.predict(Xtr[ytr == ib][:500].astype(np.float32), verbose=0)
    ca, cb = ea.mean(0), eb.mean(0)
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-8))
    spread = float((np.linalg.norm(ea - ca, axis=1).mean() + np.linalg.norm(eb - cb, axis=1).mean()) / 2)
    separation = float(np.linalg.norm(ca - cb) / (spread + 1e-8))  # >1: centroids further apart than spread

    E = np.vstack([ea, eb])
    E = E - E.mean(0)
    _, _, vt = np.linalg.svd(E, full_matrices=False)
    P = E @ vt[:2].T
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(P[:len(ea), 0], P[:len(ea), 1], s=6, alpha=0.5, label=A)
    ax.scatter(P[len(ea):, 0], P[len(ea):, 1], s=6, alpha=0.5, label=B)
    ax.set_title(f"Embedding PCA (train): {A} vs {B}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "embedding_pca.png"), dpi=130)
    plt.close(fig)

    # descriptor histograms
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for k in range(3):
        axes[k].hist(ma[:, k], bins=40, alpha=0.55, density=True, label=A)
        axes[k].hist(mb[:, k], bins=40, alpha=0.55, density=True, label=B)
        axes[k].set_title(f"Mean channel {k + 1}  (BC={colour_overlap[channel_names[k]]:.2f})", fontsize=9)
    axes[3].hist(ta, bins=40, alpha=0.55, density=True, label=A)
    axes[3].hist(tb, bins=40, alpha=0.55, density=True, label=B)
    axes[3].set_title(f"Edge density  (BC={texture_overlap:.2f})", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "input_statistics.png"), dpi=130)
    plt.close(fig)

    wrong_conf = preds.set_index("id").loc[np.array(ids)[sel["a_as_b"] | sel["b_as_a"]], "confidence"]
    stats = {
        "class_a": a, "class_b": b,
        "n_a_as_b": int(sel["a_as_b"].sum()), "n_b_as_a": int(sel["b_as_a"].sum()),
        "mean_conf_wrong": float(wrong_conf.mean()) if len(wrong_conf) else None,
        "focus_correct": float(focus_c.mean()) if len(focus_c) else None,
        "focus_wrong": float(focus_w.mean()) if len(focus_w) else None,
        "colour_overlap_mean": float(np.mean(list(colour_overlap.values()))),
        **{f"colour_overlap_{k}": v for k, v in colour_overlap.items()},
        "texture_overlap": texture_overlap,
        "embed_centroid_cosine": cos,
        "embed_separation": separation,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--pair", nargs=2, metavar=("CLASS_A", "CLASS_B"))
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    with open(os.path.join(args.run, "run_config.json")) as f:
        run_cfg = json.load(f)
    preds_path = os.path.join(args.run, "predictions.csv")
    if not os.path.isfile(preds_path):
        raise SystemExit(f"Run `python evaluate.py --run {args.run}` first.")
    preds = pd.read_csv(preds_path)

    if args.pair:
        for c in args.pair:
            if c not in config.CLASS_NAMES:
                raise SystemExit(f"Unknown class {c}. Choose from: {', '.join(config.CLASS_NAMES)}")
        pairs = [tuple(args.pair)]
    else:
        cp = pd.read_csv(os.path.join(args.run, "confused_pairs.csv")).head(args.top)
        pairs = list(zip(cp.class_a, cp.class_b))

    arrays, _, _ = data.prepare(run_cfg["composite"], args.data_root)
    model = load(os.path.join(args.run, "best_model.keras"))
    grad_model = make_gradcam_model(model, last_conv_layer_name(run_cfg["backbone"]))
    embed_model = keras.Model(model.inputs, model.get_layer("gap").output)

    rows = []
    for a, b in pairs:
        print(f"Analysing {a} vs {b} ...")
        out_dir = os.path.join(args.run, "confusability", f"{a}__{b}")
        s = analyse_pair(a, b, model, grad_model, embed_model, arrays, preds, out_dir)
        rows.append(s)
        print(json.dumps(s, indent=2))

    summary_path = os.path.join(args.run, "confusability", "summary.csv")
    new = pd.DataFrame(rows)
    if os.path.isfile(summary_path):
        old = pd.read_csv(summary_path)
        old = old[~old.set_index(["class_a", "class_b"]).index.isin(new.set_index(["class_a", "class_b"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
