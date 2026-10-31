"""Land-use change detection by post-classification comparison.

Take two co-registered images of the same area from different dates, cut each
into 64x64 tiles (= 640 m x 640 m at Sentinel-2's 10 m resolution, exactly a
EuroSAT patch), classify every tile, and compare the two land-use maps.

A tile is reported as CHANGED only when the labels differ AND the model is
confident (>= --threshold) on both dates. Labels that differ with low
confidence are reported as UNCERTAIN instead. This matters: with ~97% per-tile
accuracy, a naive "labels differ" rule flags about 6% of tiles as changed even
when nothing changed at all (0.97 * 0.97 = 0.94). The threshold trades a bit of
recall for a large drop in false alarms -- and the synthetic benchmark below
measures that trade-off instead of guessing it.

Two modes:

  Real imagery (e.g. two Sentinel-2 true-colour exports from Copernicus Browser):
    python change_detection.py --run outputs/rgb_mobilenetv2 \
        --before kochi_2017.png --after kochi_2024.png --gsd 10

  Synthetic benchmark with known ground truth (built from TEST patches only):
    python change_detection.py --run outputs/rgb_mobilenetv2 --synthetic --scenes 50

Important for real imagery: the model was trained at 10 m/px. If your export
has a different pixel size, pass --gsd so it is resampled; otherwise a 64 px
tile no longer covers 640 m and accuracy collapses.
"""

import argparse
import json
import os

import numpy as np

import config

UNCHANGED, CHANGED, UNCERTAIN = 0, 1, 2


# ---------------------------------------------------------------- core
def tile_image(img):
    """Crop to a multiple of 64 and cut into tiles. Returns tiles [R*C,64,64,3], (R, C)."""
    p = config.PATCH_PX
    rows, cols = img.shape[0] // p, img.shape[1] // p
    if rows == 0 or cols == 0:
        raise ValueError(f"Image {img.shape[:2]} is smaller than one {p}x{p} tile.")
    img = img[:rows * p, :cols * p, :3]
    tiles = img.reshape(rows, p, cols, p, 3).swapaxes(1, 2).reshape(rows * cols, p, p, 3)
    return tiles, (rows, cols)


def classify_tiles(model, tiles, tta=True, batch=256):
    X = tiles.astype(np.float32)
    if not tta:
        return model.predict(X, batch_size=batch, verbose=0)
    probs = 0
    for k in range(4):
        r = np.rot90(X, k, axes=(1, 2))
        probs = probs + model.predict(np.ascontiguousarray(r), batch_size=batch, verbose=0)
        probs = probs + model.predict(np.ascontiguousarray(r[:, :, ::-1]), batch_size=batch, verbose=0)
    return probs / 8


def detect_change(model, before, after, threshold=0.7, tta=True):
    """before/after: uint8 HxWx3 arrays, same size, 10 m/px. Returns a result dict."""
    if before.shape[:2] != after.shape[:2]:
        raise ValueError(f"Images must be the same size and co-registered: {before.shape} vs {after.shape}")
    tb, (R, C) = tile_image(before)
    ta, _ = tile_image(after)
    pb = classify_tiles(model, tb, tta)
    pa = classify_tiles(model, ta, tta)
    lb, la = pb.argmax(1), pa.argmax(1)
    cb, ca = pb.max(1), pa.max(1)

    status = np.full(len(lb), UNCHANGED)
    differs = lb != la
    confident = (cb >= threshold) & (ca >= threshold)
    status[differs & confident] = CHANGED
    status[differs & ~confident] = UNCERTAIN

    n = len(config.CLASS_NAMES)
    transitions = np.zeros((n, n), dtype=int)
    for i, j in zip(lb[status == CHANGED], la[status == CHANGED]):
        transitions[i, j] += 1

    return {
        "grid": (R, C),
        "labels_before": lb.reshape(R, C), "labels_after": la.reshape(R, C),
        "conf_before": cb.reshape(R, C), "conf_after": ca.reshape(R, C),
        "status": status.reshape(R, C),
        "transitions": transitions,
        "threshold": threshold,
    }


def summarise(result):
    """JSON-friendly summary: area per class on each date, and from->to changes in hectares."""
    ha = config.PATCH_AREA_HA
    lb, la, st = result["labels_before"], result["labels_after"], result["status"]
    area_before = {c: float((lb == k).sum() * ha) for k, c in enumerate(config.CLASS_NAMES)}
    area_after = {c: float((la == k).sum() * ha) for k, c in enumerate(config.CLASS_NAMES)}
    tr = result["transitions"]
    changes = []
    for i in range(tr.shape[0]):
        for j in range(tr.shape[1]):
            if tr[i, j]:
                changes.append({"from": config.CLASS_NAMES[i], "to": config.CLASS_NAMES[j],
                                "tiles": int(tr[i, j]), "hectares": float(tr[i, j] * ha)})
    changes.sort(key=lambda d: -d["tiles"])
    return {
        "grid_rows": int(result["grid"][0]), "grid_cols": int(result["grid"][1]),
        "tile_area_ha": ha, "threshold": result["threshold"],
        "tiles_total": int(st.size), "tiles_changed": int((st == CHANGED).sum()),
        "tiles_uncertain": int((st == UNCERTAIN).sum()),
        "area_changed_ha": float((st == CHANGED).sum() * ha),
        "area_before_ha": area_before, "area_after_ha": area_after,
        "net_change_ha": {c: area_after[c] - area_before[c] for c in config.CLASS_NAMES},
        "transitions": changes,
    }


# ---------------------------------------------------------------- rendering
def label_map_image(labels):
    p = config.PATCH_PX
    cols = np.array([config.CLASS_COLORS[c] for c in config.CLASS_NAMES], dtype=np.uint8)
    return np.repeat(np.repeat(cols[labels], p, axis=0), p, axis=1)


def change_overlay(after, status):
    """Darken unchanged tiles, outline changed tiles red and uncertain tiles amber."""
    p = config.PATCH_PX
    R, C = status.shape
    img = after[:R * p, :C * p, :3].astype(np.float32).copy()
    for r in range(R):
        for c in range(C):
            ys, xs = slice(r * p, (r + 1) * p), slice(c * p, (c + 1) * p)
            if status[r, c] == UNCHANGED:
                img[ys, xs] *= 0.45
            else:
                color = (235, 50, 40) if status[r, c] == CHANGED else (240, 180, 40)
                img[ys, xs] = img[ys, xs] * 0.75 + np.array(color) * 0.25
                for t in range(3):
                    img[r * p + t, xs] = color
                    img[(r + 1) * p - 1 - t, xs] = color
                    img[ys, c * p + t] = color
                    img[ys, (c + 1) * p - 1 - t] = color
    return img.clip(0, 255).astype(np.uint8)


def render_figure(before, after, result, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    p = config.PATCH_PX
    R, C = result["grid"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    panels = [(before[:R * p, :C * p], "Before"), (after[:R * p, :C * p], "After"),
              (label_map_image(result["labels_before"]), "Land use (before)"),
              (label_map_image(result["labels_after"]), "Land use (after)"),
              (change_overlay(after, result["status"]), "Change (red) / uncertain (amber)")]
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(t)
        ax.axis("off")
    used = np.unique(np.concatenate([result["labels_before"].ravel(), result["labels_after"].ravel()]))
    handles = [Patch(color=np.array(config.CLASS_COLORS[config.CLASS_NAMES[k]]) / 255,
                     label=config.PRETTY_NAMES[config.CLASS_NAMES[k]]) for k in used]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_image(path, gsd=10.0):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if abs(gsd - config.METRES_PER_PX) > 1e-6:
        scale = gsd / config.METRES_PER_PX
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    return np.asarray(img)


# ---------------------------------------------------------------- synthetic benchmark
def make_synthetic_pair(X, y, rng, grid=8, change_frac=0.25):
    """Build a before/after scene from real test patches with KNOWN changes.

    Before: a grid of random patches. After: the same scene, but for unchanged
    cells a DIFFERENT patch of the same class is used (so the classifier sees
    new pixels on both dates, like real imagery with seasonal/lighting
    differences) and for changed cells a patch of a different class.
    """
    p = config.PATCH_PX
    by_class = [np.where(y == k)[0] for k in range(len(config.CLASS_NAMES))]
    n = grid * grid
    cls_before = rng.integers(0, len(by_class), n)
    changed = rng.random(n) < change_frac
    cls_after = cls_before.copy()
    for i in np.where(changed)[0]:
        choices = [k for k in range(len(by_class)) if k != cls_before[i]]
        cls_after[i] = rng.choice(choices)
    before = np.zeros((grid * p, grid * p, 3), dtype=np.uint8)
    after = np.zeros_like(before)
    for i in range(n):
        r, c = divmod(i, grid)
        a, b = rng.choice(by_class[cls_before[i]], 2, replace=False)
        before[r * p:(r + 1) * p, c * p:(c + 1) * p] = X[a]
        after[r * p:(r + 1) * p, c * p:(c + 1) * p] = X[rng.choice(by_class[cls_after[i]])] if changed[i] else X[b]
    return before, after, changed.reshape(grid, grid)


def run_synthetic(model, X, y, out_dir, scenes=50, grid=8, change_frac=0.25,
                  thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), tta=True):
    rng = np.random.default_rng(config.SEED)
    pairs = [make_synthetic_pair(X, y, rng, grid, change_frac) for _ in range(scenes)]
    # Classify once, then sweep the threshold on the cached probabilities.
    cached = []
    for before, after, gt in pairs:
        tb, _ = tile_image(before)
        ta, _ = tile_image(after)
        cached.append((classify_tiles(model, tb, tta), classify_tiles(model, ta, tta), gt.ravel()))

    rows = []
    for th in thresholds:
        tp = fp = fn = tn = unc = 0
        for pb, pa, gt in cached:
            differs = pb.argmax(1) != pa.argmax(1)
            conf = (pb.max(1) >= th) & (pa.max(1) >= th)
            flagged = differs & conf
            unc += int((differs & ~conf).sum())
            tp += int((flagged & gt).sum())
            fp += int((flagged & ~gt).sum())
            fn += int((~flagged & gt).sum())
            tn += int((~flagged & ~gt).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        rows.append({"threshold": th, "precision": prec, "recall": rec, "f1": f1,
                     "false_alarm_rate": fp / max(fp + tn, 1), "uncertain_tiles": unc,
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn})

    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "synthetic_benchmark.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in ("precision", "recall", "f1", "false_alarm_rate"):
        ax.plot(df.threshold, df[col], marker="o", label=col.replace("_", " "))
    ax.set_xlabel("Confidence threshold (both dates)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(f"Synthetic change benchmark ({scenes} scenes, {grid}x{grid} tiles, {int(change_frac * 100)}% changed)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "synthetic_benchmark.png"), dpi=130)
    plt.close(fig)

    before, after, gt = pairs[0]
    res = detect_change(model, before, after, threshold=0.7, tta=tta)
    render_figure(before, after, res, os.path.join(out_dir, "synthetic_example.png"),
                  f"Synthetic scene: {int(gt.sum())} true changes, "
                  f"{int((res['status'] == CHANGED).sum())} detected")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--before")
    p.add_argument("--after")
    p.add_argument("--gsd", type=float, default=10.0, help="Metres per pixel of the input images.")
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--scenes", type=int, default=50)
    p.add_argument("--grid", type=int, default=8)
    p.add_argument("--change-frac", type=float, default=0.25)
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    from model import load
    model = load(os.path.join(args.run, "best_model.keras"))
    out_dir = args.out or os.path.join(args.run, "change_detection")
    os.makedirs(out_dir, exist_ok=True)

    if args.synthetic:
        import data
        with open(os.path.join(args.run, "run_config.json")) as f:
            composite = json.load(f)["composite"]
        arrays, _, _ = data.prepare(composite, args.data_root)
        X, y, _ = arrays["test"]
        df = run_synthetic(model, X, y, out_dir, args.scenes, args.grid, args.change_frac, tta=not args.no_tta)
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nSaved to {out_dir}")
        return

    if not (args.before and args.after):
        raise SystemExit("Pass --before and --after images, or --synthetic.")
    before = load_image(args.before, args.gsd)
    after = load_image(args.after, args.gsd)
    if before.shape != after.shape:
        print(f"Warning: sizes differ {before.shape} vs {after.shape}; cropping to the overlap. "
              "Make sure both exports cover exactly the same extent.")
        h, w = min(before.shape[0], after.shape[0]), min(before.shape[1], after.shape[1])
        before, after = before[:h, :w], after[:h, :w]
    res = detect_change(model, before, after, threshold=args.threshold, tta=not args.no_tta)
    summary = summarise(res)
    stem = f"{os.path.splitext(os.path.basename(args.before))[0]}_vs_{os.path.splitext(os.path.basename(args.after))[0]}"
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    render_figure(before, after, res, os.path.join(out_dir, f"{stem}.png"), stem)
    print(json.dumps({k: summary[k] for k in ("tiles_total", "tiles_changed", "tiles_uncertain",
                                              "area_changed_ha")}, indent=2))
    for t in summary["transitions"][:10]:
        print(f"  {t['from']:>20s} -> {t['to']:<20s} {t['hectares']:8.1f} ha ({t['tiles']} tiles)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
