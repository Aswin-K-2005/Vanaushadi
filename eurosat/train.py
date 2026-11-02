"""Transfer-learning training on EuroSAT.

Two phases (same recipe as the plant model, so it should feel familiar):
  1. Backbone frozen, train only the new classifier head.
  2. Unfreeze the top of the backbone and fine-tune with a ~100x smaller LR.

Augmentation uses the dihedral group (flips + 90-degree rotations) only.
Satellite images have no "up", so all 8 orientations are equally valid views
of the same ground. Colour jitter is deliberately NOT used: spectral colour is
the signal that separates crop types, and the confusability study would be
muddied if we trained the model to ignore it.

Examples:
  python train.py                                  # RGB, MobileNetV2, defaults
  python train.py --composite ms_falsecolor        # NIR-R-G composite
  python train.py --backbone efficientnetb0 --img-size 224
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

import config
import data
from model import build_model, set_finetune

import keras


def augment(x, y):
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    x = tf.image.random_flip_left_right(x)
    return x, y


def make_ds(X, y, batch, training):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=config.SEED, reshuffle_each_iteration=True).map(
            augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(lambda a, b: (tf.cast(a, tf.float32), b)).batch(batch).prefetch(tf.data.AUTOTUNE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--composite", default="rgb", choices=list(config.COMPOSITES))
    p.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0", "resnet50"])
    p.add_argument("--img-size", type=int, default=224,
                   help="64px patches are upsampled to this. 224 gives a 7x7 Grad-CAM grid; 64 would give 2x2.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--ft-layers", type=int, default=60, help="How many backbone layers to unfreeze in phase 2.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-5,
                   help="Fine-tuning LR. Keep it small: large updates wreck the pretrained filters.")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (from scratch, for an ablation).")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Use only this fraction of the training set (label-efficiency experiment).")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None, help="Output dir (default outputs/<composite>_<backbone>)")
    args = p.parse_args()

    keras.utils.set_random_seed(config.SEED)
    out_dir = args.out or config.run_dir(args.composite, args.backbone)
    os.makedirs(out_dir, exist_ok=True)

    arrays, _, _ = data.prepare(args.composite, args.data_root)
    Xtr, ytr, _ = arrays["train"]
    Xva, yva, _ = arrays["val"]
    if args.train_fraction < 1.0:
        rng = np.random.default_rng(config.SEED)
        keep = np.concatenate([
            rng.permutation(np.where(ytr == c)[0])[:max(1, int(args.train_fraction * (ytr == c).sum()))]
            for c in range(len(config.CLASS_NAMES))
        ])
        Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"train={len(Xtr)}  val={len(Xva)}  composite={args.composite}  backbone={args.backbone}")

    train_ds = make_ds(Xtr, ytr, args.batch, training=True)
    val_ds = make_ds(Xva, yva, args.batch, training=False)

    weights = None if args.weights.lower() == "none" else args.weights
    model, base = build_model(args.backbone, args.img_size, len(config.CLASS_NAMES), weights=weights)
    best_path = os.path.join(out_dir, "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv"), append=True),
    ]

    history = {}

    def run(epochs, lr, phase):
        if epochs <= 0:
            return
        model.compile(optimizer=keras.optimizers.Adam(lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cbs = callbacks + [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
                           keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                                         restore_best_weights=True)]
        h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(a) for a in v)
        history.setdefault("phase", []).extend([phase] * len(h.history["loss"]))

    if os.path.isfile(os.path.join(out_dir, "history.csv")):
        os.remove(os.path.join(out_dir, "history.csv"))

    t0 = time.time()
    if weights is None:
        # From scratch: everything trainable from the start, single phase.
        set_finetune(model, base, n_unfrozen=len(base.layers))
        run(args.head_epochs + args.ft_epochs, args.lr, "scratch")
    else:
        run(args.head_epochs, args.lr, "head")
        set_finetune(model, base, args.ft_layers)
        run(args.ft_epochs, args.ft_lr, "finetune")
    elapsed = time.time() - t0

    # ModelCheckpoint has the best epoch across both phases on disk.
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "class_names": config.CLASS_NAMES, "n_train": int(len(Xtr)),
                   "train_seconds": round(elapsed, 1), "history": history}, f, indent=2)
    print(f"Done in {elapsed / 60:.1f} min. Best model: {best_path}")
    print(f"Next: python evaluate.py --run {out_dir}")


if __name__ == "__main__":
    main()
