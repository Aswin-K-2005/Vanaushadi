"""Compare several evaluated runs side by side: the results table for the report.

For each run: overall accuracy, macro-F1, and the confusion rate of the pairs
that were hardest for the FIRST run (your baseline). The question this answers
is e.g. "does the NIR false-colour composite reduce AnnualCrop<->PermanentCrop
confusion compared with true colour, using the identical pipeline and split?"

Usage:
  python compare.py outputs/ms_rgb_mobilenetv2 outputs/ms_falsecolor_mobilenetv2 outputs/ms_swir_mobilenetv2
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import config


def load_run(run, tta):
    suffix = "_tta" if tta else ""
    with open(os.path.join(run, f"metrics{suffix}.json")) as f:
        m = json.load(f)
    cm = np.array(m["confusion_matrix"], dtype=float)
    return m, cm / np.maximum(cm.sum(1, keepdims=True), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--out", default=os.path.join(config.OUTPUT_ROOT, "comparison.csv"))
    args = p.parse_args()

    loaded = [(r, *load_run(r, args.tta)) for r in args.runs]
    base_cm = loaded[0][2]
    n = len(config.CLASS_NAMES)
    pairs = sorted(((base_cm[i, j] + base_cm[j, i]) / 2, i, j) for i in range(n) for j in range(i + 1, n))[::-1]
    pairs = pairs[:args.top]

    rows = []
    for run, m, cmn in loaded:
        row = {"run": os.path.basename(os.path.normpath(run)),
               "accuracy": round(m["accuracy"] * 100, 2), "macro_f1": round(m["macro_f1"] * 100, 2)}
        for _, i, j in pairs:
            key = f"{config.CLASS_NAMES[i]}<->{config.CLASS_NAMES[j]}"
            row[key] = round((cmn[i, j] + cmn[j, i]) / 2 * 100, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"\n(pair columns = symmetric confusion %, lower is better)  saved to {args.out}")


if __name__ == "__main__":
    main()
