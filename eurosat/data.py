"""Dataset download, the fixed train/val/test split, and loading images into arrays.

Design choices worth mentioning in the report:
  * The split is made ONCE (stratified per class, fixed seed) and written to
    outputs/split.csv. Every experiment -- RGB, false-colour, every backbone --
    reads the same file, so differences between runs come from the model and
    the input bands, never from a luckier test set.
  * Samples are identified by "Class/Class_N" without the file extension. The
    RGB release (.jpg) and the multispectral release (.tif) use the same
    names, so one split covers both.
  * The whole dataset is small (27,000 x 64 x 64 x 3 bytes = ~330 MB), so it is
    decoded once into a uint8 .npy cache and fed from memory. That makes each
    epoch fast and keeps the tf.data pipeline trivial.
"""

import argparse
import json
import os
import zipfile

import numpy as np
import pandas as pd

import config

IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# ---------------------------------------------------------------- download
def _download(urls, dest_zip):
    import urllib.request

    last_err = None
    for url in urls:
        try:
            print(f"Downloading {url}\n  -> {dest_zip}")
            urllib.request.urlretrieve(url, dest_zip)
            return
        except Exception as e:  # try the next mirror
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(
        f"Could not download from any mirror ({last_err}). Download the zip "
        f"manually, unzip it under {config.DATA_ROOT}, and re-run."
    )


def find_class_root(search_dir):
    """Return the folder whose sub-folders are the 10 class names.

    The zips nest differently (EuroSAT_RGB/, 2750/, ds/images/remote_sensing/...),
    so we search instead of hard-coding a path.
    """
    wanted = set(config.CLASS_NAMES)
    for root, dirs, _ in os.walk(search_dir):
        if wanted.issubset(set(dirs)):
            return root
    return None


def ensure_dataset(kind="rgb", data_root=None):
    """Make sure the RGB ('rgb') or multispectral ('ms') release is on disk."""
    data_root = data_root or config.DATA_ROOT
    target = os.path.join(data_root, "EuroSAT_RGB" if kind == "rgb" else "EuroSAT_MS")
    root = find_class_root(target) if os.path.isdir(target) else None
    if root:
        return root

    os.makedirs(target, exist_ok=True)
    zip_path = target + ".zip"
    if not os.path.isfile(zip_path):
        _download(config.RGB_URLS if kind == "rgb" else config.MS_URLS, zip_path)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    root = find_class_root(target)
    if not root:
        raise RuntimeError(f"Extracted {zip_path} but found no class folders inside {target}")
    return root


# ---------------------------------------------------------------- split
def list_samples(class_root):
    rows = []
    for label, cls in enumerate(config.CLASS_NAMES):
        cls_dir = os.path.join(class_root, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(IMG_EXTS):
                rows.append({"id": f"{cls}/{os.path.splitext(fname)[0]}", "label": label, "cls": cls})
    return pd.DataFrame(rows)


def make_split(class_root, out_csv, seed=config.SEED):
    """Stratified 70/15/15 split, written to CSV. Never overwrites an existing split."""
    if os.path.isfile(out_csv):
        return pd.read_csv(out_csv)

    df = list_samples(class_root)
    rng = np.random.default_rng(seed)
    f_train, f_val, _ = config.SPLIT_FRACTIONS
    parts = []
    for _, group in df.groupby("label"):
        idx = rng.permutation(len(group))
        n_train = int(round(f_train * len(group)))
        n_val = int(round(f_val * len(group)))
        split = np.array(["test"] * len(group), dtype=object)
        split[idx[:n_train]] = "train"
        split[idx[n_train:n_train + n_val]] = "val"
        g = group.copy()
        g["split"] = split
        parts.append(g)
    out = pd.concat(parts).sort_values("id").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote split to {out_csv}:")
    print(out.groupby(["cls", "split"]).size().unstack())
    return out


# ---------------------------------------------------------------- image loading
def _resolve(class_root, sample_id):
    base = os.path.join(class_root, sample_id)
    for ext in IMG_EXTS:
        if os.path.isfile(base + ext):
            return base + ext
    raise FileNotFoundError(base + ".*")


def _read_ms(path):
    import tifffile

    arr = tifffile.imread(path)  # (64, 64, 13) or (13, 64, 64)
    if arr.ndim == 3 and arr.shape[0] == len(config.MS_BANDS) and arr.shape[-1] != len(config.MS_BANDS):
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def compute_band_stretch(class_root, ids, bands, n=2000, seed=config.SEED):
    """Per-band 2nd/98th percentile from TRAINING images only (no test leakage).

    Sentinel-2 values are reflectance x 10000 and have very different ranges
    per band (NIR is much brighter than blue), so each band gets its own
    linear stretch to 0..255.
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    band_idx = [config.MS_BANDS.index(b) for b in bands]
    vals = [[] for _ in bands]
    for sid in pick:
        arr = _read_ms(_resolve(class_root, sid))
        for k, bi in enumerate(band_idx):
            vals[k].append(arr[..., bi].ravel()[::8])
    stretch = {}
    for k, b in enumerate(bands):
        v = np.concatenate(vals[k])
        stretch[b] = [float(np.percentile(v, 2)), float(np.percentile(v, 98))]
    return stretch


def ms_to_composite(arr, bands, stretch):
    out = np.empty(arr.shape[:2] + (3,), dtype=np.uint8)
    for k, b in enumerate(bands):
        lo, hi = stretch[b]
        x = (arr[..., config.MS_BANDS.index(b)] - lo) / max(hi - lo, 1e-6)
        out[..., k] = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return out


def load_arrays(composite, split_df, class_root, cache_dir):
    """Return {split: (X uint8 [N,64,64,3], y int [N], ids [N])}, cached as .npy."""
    from PIL import Image

    os.makedirs(cache_dir, exist_ok=True)
    x_path = os.path.join(cache_dir, f"X_{composite}.npy")
    meta_path = os.path.join(cache_dir, f"ids_{composite}.json")

    if os.path.isfile(x_path) and os.path.isfile(meta_path):
        X = np.load(x_path, mmap_mode="r")
        with open(meta_path) as f:
            ids = json.load(f)["ids"]
    else:
        ids = split_df["id"].tolist()
        X = np.empty((len(ids), config.PATCH_PX, config.PATCH_PX, 3), dtype=np.uint8)
        bands = config.COMPOSITES[composite]
        stretch = None
        if bands is not None:
            train_ids = split_df.loc[split_df.split == "train", "id"].tolist()
            stretch = compute_band_stretch(class_root, train_ids, bands)
        print(f"Decoding {len(ids)} images for composite '{composite}' (one-time) ...")
        for i, sid in enumerate(ids):
            path = _resolve(class_root, sid)
            if bands is None:
                img = Image.open(path).convert("RGB")
                if img.size != (config.PATCH_PX, config.PATCH_PX):
                    img = img.resize((config.PATCH_PX, config.PATCH_PX))
                X[i] = np.asarray(img)
            else:
                X[i] = ms_to_composite(_read_ms(path), bands, stretch)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(ids)}")
        np.save(x_path, X)
        with open(meta_path, "w") as f:
            json.dump({"ids": ids, "stretch": stretch}, f)

    index = {sid: i for i, sid in enumerate(ids)}
    out = {}
    for split in ("train", "val", "test"):
        sub = split_df[split_df.split == split]
        rows = np.array([index[s] for s in sub["id"]])
        out[split] = (np.asarray(X[rows]), sub["label"].to_numpy(), sub["id"].tolist())
    return out


def prepare(composite, data_root=None, output_root=None):
    """One call that does download -> split -> arrays. Used by train and evaluate."""
    output_root = output_root or config.OUTPUT_ROOT
    kind = "rgb" if composite == "rgb" else "ms"
    class_root = ensure_dataset(kind, data_root)
    split_csv = os.path.join(output_root, "split.csv")
    split_df = make_split(class_root, split_csv)
    arrays = load_arrays(composite, split_df, class_root, os.path.join(output_root, "cache"))
    return arrays, split_df, class_root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download EuroSAT and build the fixed split + array cache.")
    p.add_argument("--composite", default="rgb", choices=list(config.COMPOSITES))
    p.add_argument("--data-root", default=None)
    args = p.parse_args()
    arrays, _, root = prepare(args.composite, args.data_root)
    print(f"Class root: {root}")
    for k, (X, y, _) in arrays.items():
        print(f"{k:5s} X={X.shape} y={y.shape}")
