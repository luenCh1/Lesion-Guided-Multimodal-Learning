"""Metrics report for binary classification predictions.

Reads one or more prediction CSVs (columns: sample_id, y, p) and reports
image-level AUC, accuracy, sensitivity, specificity, PPV, NPV and F1 at a
fixed threshold, with bootstrap confidence intervals.

Usage
-----
python metrics_report.py --pred model_a=preds_a.csv --pred model_b=preds_b.csv \
    --threshold 0.5 --bootstrap 2000 --out metrics.csv

Every count is derived from the input files; nothing about cohort size is
hard-coded.
"""
import argparse
import csv

import numpy as np


def auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def point_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return dict(
        n=len(y), positive=int((y == 1).sum()), negative=int((y == 0).sum()),
        TP=tp, TN=tn, FP=fp, FN=fn,
        AUC=auc(y, p),
        ACC=(tp + tn) / len(y),
        Sensitivity=tp / (tp + fn) if (tp + fn) else float("nan"),
        Specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        PPV=tp / (tp + fp) if (tp + fp) else float("nan"),
        NPV=tn / (tn + fn) if (tn + fn) else float("nan"),
        F1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
    )


def bootstrap_ci(y, p, thr, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        stats.append(point_metrics(y[idx], p[idx], thr))
    keys = ["AUC", "ACC", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in stats], dtype=float)
        out[k + "_CI_low"], out[k + "_CI_high"] = np.nanpercentile(v, [2.5, 97.5])
    return out


def read_preds(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return np.array([int(r["y"]) for r in rows]), np.array([float(r["p"]) for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path.csv (repeatable)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="metrics.csv")
    a = ap.parse_args()

    results = []
    for item in a.pred:
        name, path = item.split("=", 1)
        y, p = read_preds(path)
        r = point_metrics(y, p, a.threshold)
        r.update(bootstrap_ci(y, p, a.threshold, B=a.bootstrap))
        r["model"] = name
        results.append(r)

    cols = ["model", "n", "positive", "negative", "TP", "TN", "FP", "FN",
            "AUC", "ACC", "Sensitivity", "Specificity", "PPV", "NPV", "F1",
            "AUC_CI_low", "AUC_CI_high", "ACC_CI_low", "ACC_CI_high",
            "Sensitivity_CI_low", "Sensitivity_CI_high",
            "Specificity_CI_low", "Specificity_CI_high",
            "PPV_CI_low", "PPV_CI_high", "NPV_CI_low", "NPV_CI_high",
            "F1_CI_low", "F1_CI_high"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("wrote %s (%d models)" % (a.out, len(results)))


if __name__ == "__main__":
    main()
