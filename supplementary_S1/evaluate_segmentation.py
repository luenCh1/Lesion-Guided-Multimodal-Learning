#!/usr/bin/env python3
"""
evaluate_segmentation.py -- deterministic, dependency-light evaluation of binary
lesion-segmentation predictions against reference masks.

Design goals
------------
1. One script, one code path: the same masks produce byte-identical metrics on
   every run (sorted file order, fixed bootstrap seed, no parallelism).
2. Explicit aggregation. Every summary is labelled as one of:
      macro_image   : metric computed per image, then unweighted mean over images
      micro_pooled  : TP/FP/FN pooled over all pixels of all images, metric then computed once
      patient_mean  : metric computed per image, averaged within patient, then mean over patients
   Macro and micro values are NOT interchangeable and are reported side by side.
3. Metrics that are mathematically constrained to agree do agree, because they
   are all derived from the same TP/FP/FN counts per image:
      precision = TP/(TP+FP); recall = TP/(TP+FN)
      Dice      = 2TP/(2TP+FP+FN) = 2PR/(P+R)
      IoU       = TP/(TP+FP+FN)   = PR/(P+R-PR)
   Hence IoU <= min(precision, recall) for every image and for every mean of images.
4. Boundary-sensitive metrics: HD95 and ASSD, reported in pixels. Physical units
   (mm) are only meaningful when pixel spacing at the lesion is known, which it
   generally is not for zoom-endoscopy frames; the script therefore does not
   invent a mm conversion.
5. Full provenance: SHA-256 of every input mask plus the script itself, the
   exact arguments, library versions, and UTC timestamp are written to
   provenance.json, so a reported number can be traced to a mask set.

Usage
-----
    python evaluate_segmentation.py --pred-dir preds --ref-dir refs --out-dir out
    python evaluate_segmentation.py --pred-dir preds --ref-dir refs --out-dir out \
        --patient-map patients.csv --bootstrap 2000 --seed 20260922

Mask conventions
----------------
* PNG/TIF/BMP, any bit depth; values are normalised to [0,1] by dividing by the
  observed maximum when max > 1, then binarised at --threshold (default 0.5).
* File stems must match between --pred-dir and --ref-dir.
* Optional --patient-map CSV with columns: filename,patient_id. Without it,
  every image is treated as its own patient.

Outputs (in --out-dir)
----------------------
    metrics_per_image.csv   one row per image, all raw metrics
    metrics_summary.json    all aggregations with bootstrap CIs
    summary_table.md        publication-ready summary table
    provenance.json         hashes, arguments, environment
    missing_or_invalid.txt  any mask pairs that could not be evaluated

Reference: this script implements the metric definitions requested in the
review (Dice, IoU/Jaccard, precision, recall, HD95, ASSD) with the aggregation
level stated for each number.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_VERSION = "1.1.0"
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mask(path: Path, threshold: float) -> np.ndarray:
    """Load a mask as a boolean array. Raises ValueError on unusable input."""
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        # Collapse RGB/RGBA masks only when all channels agree; otherwise refuse,
        # because silently taking one channel would change the mask.
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if not (np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 0], arr[:, :, 2])):
            raise ValueError(f"{path.name}: colour image is not a single-channel mask")
        arr = arr[:, :, 0]
    arr = arr.astype(np.float64)
    if arr.size == 0:
        raise ValueError(f"{path.name}: empty image")
    if arr.max() > 1.0:
        arr = arr / arr.max()
    mask = arr >= threshold
    if mask.all():
        raise ValueError(f"{path.name}: mask covers the entire image (check polarity/inversion)")
    return mask


def find_masks(directory: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            key = p.stem
            if key in found:
                raise SystemExit(f"duplicate stem '{key}' in {directory}")
            found[key] = p
    if not found:
        raise SystemExit(f"no mask images in {directory}")
    return found


# --------------------------------------------------------------------------- #
# Segmentation metrics
# --------------------------------------------------------------------------- #
def confusion(pred: np.ndarray, ref: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    return tp, fp, fn, tn


def safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else num / den


def surface(mask: np.ndarray) -> np.ndarray:
    """4-connected boundary pixels of a binary mask."""
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = (
        p[1:-1, 1:-1]
        & p[:-2, 1:-1]
        & p[2:, 1:-1]
        & p[1:-1, :-2]
        & p[1:-1, 2:]
    )
    return np.argwhere(mask & ~eroded).astype(np.int32)


def directed_distances(a: np.ndarray, b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For every point in a, the Euclidean distance to the nearest point in b."""
    if a.size == 0 or b.size == 0:
        return np.zeros((0,), dtype=np.float64)
    b = b.astype(np.float64)
    out = np.empty((a.shape[0],), dtype=np.float64)
    for start in range(0, a.shape[0], chunk):
        block = a[start : start + chunk].astype(np.float64)
        d = block[:, None, :] - b[None, :, :]
        out[start : start + block.shape[0]] = np.sqrt((d * d).sum(axis=2)).min(axis=1)
    return out


def hd95_assd(pred: np.ndarray, ref: np.ndarray) -> Tuple[float, float, float]:
    """Return (hd95_concat, hd95_maxdir, assd) in pixels.

    hd95_concat : 95th percentile of the union of both directed distance sets
                  (MedPy convention).
    hd95_maxdir : max of the two directional 95th percentiles (MONAI convention).
    Both are reported because the two conventions differ; the manuscript must
    name the one it uses.
    """
    sa, sb = surface(pred), surface(ref)
    if sa.size == 0 or sb.size == 0:
        return float("nan"), float("nan"), float("nan")
    d_ab = directed_distances(sa, sb)
    d_ba = directed_distances(sb, sa)
    both = np.concatenate([d_ab, d_ba])
    hd95_concat = float(np.percentile(both, 95))
    hd95_maxdir = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    assd = float(both.mean())
    return hd95_concat, hd95_maxdir, assd


def image_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    tp, fp, fn, tn = confusion(pred, ref)
    # Degenerate cases are defined explicitly rather than left as NaN, so that
    # one missed lesion cannot silently drop out of a macro average.
    if ref.sum() == 0 and pred.sum() == 0:
        dice = iou = precision = recall = specificity = 1.0
        hd95c = hd95m = assd = float("nan")
    elif pred.sum() == 0 or ref.sum() == 0:
        dice = iou = precision = recall = 0.0
        specificity = 1.0 if pred.sum() == 0 else 0.0
        hd95c = hd95m = assd = float("nan")
    else:
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        hd95c, hd95m, assd = hd95_assd(pred, ref)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "hd95_concat_px": hd95c,
        "hd95_maxdir_px": hd95m,
        "assd_px": assd,
        "ref_area_px": int(ref.sum()),
        "pred_area_px": int(pred.sum()),
    }


# --------------------------------------------------------------------------- #
# Aggregation and bootstrap
# --------------------------------------------------------------------------- #
MEAN_METRICS = ["dice", "iou", "precision", "recall", "specificity", "hd95_concat_px", "hd95_maxdir_px", "assd_px"]


def mean_ignoring_nan(values: np.ndarray) -> float:
    vals = values[~np.isnan(values)]
    return float(vals.mean()) if vals.size else float("nan")


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    lo = (1 - level) / 2 * 100
    hi = (1 + level) / 2 * 100
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


@dataclass
class Record:
    name: str
    patient: str
    metrics: Dict[str, float]


def macro_summary(records: Sequence[Record]) -> Dict[str, float]:
    out = {}
    for m in MEAN_METRICS:
        out[m] = mean_ignoring_nan(np.array([r.metrics[m] for r in records], dtype=float))
    return out


def micro_summary(records: Sequence[Record]) -> Dict[str, float]:
    tp = sum(int(r.metrics["tp"]) for r in records)
    fp = sum(int(r.metrics["fp"]) for r in records)
    fn = sum(int(r.metrics["fn"]) for r in records)
    tn = sum(int(r.metrics["tn"]) for r in records)
    return {
        "dice": safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": safe_div(tp, tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
    }


def patient_summary(records: Sequence[Record]) -> Dict[str, float]:
    groups: Dict[str, List[Record]] = {}
    for r in records:
        groups.setdefault(r.patient, []).append(r)
    out = {}
    for m in MEAN_METRICS:
        per_patient = [
            mean_ignoring_nan(np.array([r.metrics[m] for r in rs], dtype=float))
            for rs in groups.values()
        ]
        out[m] = mean_ignoring_nan(np.array(per_patient, dtype=float))
    out["n_patients"] = len(groups)
    return out


def bootstrap_macro(records: Sequence[Record], n_boot: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(records)
    matrix = {m: np.array([r.metrics[m] for r in records], dtype=float) for m in MEAN_METRICS}
    values = {m: np.empty(n_boot, dtype=float) for m in MEAN_METRICS}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in MEAN_METRICS:
            values[m][b] = mean_ignoring_nan(matrix[m][idx])
    return {m: percentile_ci(values[m]) for m in MEAN_METRICS}


def bootstrap_patient(records: Sequence[Record], n_boot: int, seed: int) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed + 1)
    groups: Dict[str, List[Record]] = {}
    for r in records:
        groups.setdefault(r.patient, []).append(r)
    keys = sorted(groups)
    per_patient = {
        m: np.array(
            [mean_ignoring_nan(np.array([r.metrics[m] for r in groups[k]], dtype=float)) for k in keys],
            dtype=float,
        )
        for m in MEAN_METRICS
    }
    n = len(keys)
    values = {m: np.empty(n_boot, dtype=float) for m in MEAN_METRICS}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for m in MEAN_METRICS:
            values[m][b] = mean_ignoring_nan(per_patient[m][idx])
    return {m: percentile_ci(values[m]) for m in MEAN_METRICS}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def fmt(value: float, ci: Optional[Tuple[float, float]] = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if ci is None:
        return f"{value:.4f}"
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return f"{value:.4f}"
    return f"{value:.4f} ({lo:.4f}-{hi:.4f})"


def write_summary_table(path: Path, per_image: Sequence[Record], macro: Dict[str, float], micro: Dict[str, float],
                        patient: Dict[str, float], boots: Dict[str, Dict[str, Tuple[float, float]]],
                        boots_patient: Dict[str, Dict[str, Tuple[float, float]]]) -> None:
    lines = []
    lines.append("# Segmentation metrics (generated by evaluate_segmentation.py)")
    lines.append("")
    lines.append(f"Images: {len(per_image)}; patients: {patient.get('n_patients', len(per_image))}; "
                 f"bootstrap resamples: {BOOTSTRAP_N}; seed: {SEED}")
    lines.append("")
    lines.append("| Metric | Macro (per image, mean) 95% CI | Micro (pooled pixels) | Patient-level mean 95% CI |")
    lines.append("|---|---|---|---|")
    for m in ["dice", "iou", "precision", "recall", "specificity"]:
        lines.append(f"| {m} | {fmt(macro[m], boots['macro'][m])} | {fmt(micro[m])} | "
                     f"{fmt(patient[m], boots_patient['patient'][m])} |")
    for m in ["hd95_concat_px", "hd95_maxdir_px", "assd_px"]:
        lines.append(f"| {m} | {fmt(macro[m], boots['macro'][m])} | NA (not poolable) | "
                     f"{fmt(patient[m], boots_patient['patient'][m])} |")
    lines.append("")
    lines.append("Macro = each image contributes equally (standard for per-image Dice/IoU tables).")
    lines.append("Micro = all pixels pooled; dominated by large lesions and therefore not comparable to macro.")
    lines.append("Patient = each patient contributes equally after within-patient averaging.")
    lines.append("HD95 is reported both as the 95th percentile of the pooled bidirectional surface distances "
                 "(hd95_concat_px, MedPy convention) and as the maximum of the two directional 95th percentiles "
                 "(hd95_maxdir_px, MONAI convention); state which one the manuscript uses.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    global BOOTSTRAP_N, SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--ref-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--patient-map", type=Path, default=None,
                    help="CSV with columns filename,patient_id (default: one patient per image)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args(argv)
    BOOTSTRAP_N, SEED = args.bootstrap, args.seed

    preds = find_masks(args.pred_dir)
    refs = find_masks(args.ref_dir)
    only_pred = sorted(set(preds) - set(refs))
    only_ref = sorted(set(refs) - set(preds))
    if only_pred or only_ref:
        raise SystemExit(f"unpaired masks: missing ref for {only_pred}; missing pred for {only_ref}")

    patient_of: Dict[str, str] = {}
    if args.patient_map:
        with open(args.patient_map, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                patient_of[row["filename"].rsplit(".", 1)[0]] = row["patient_id"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Record] = []
    invalid: List[str] = []
    hashes: Dict[str, Dict[str, str]] = {}
    for key in sorted(preds):
        try:
            pred = load_mask(preds[key], args.threshold)
            ref = load_mask(refs[key], args.threshold)
        except Exception as exc:  # noqa: BLE001 - record and continue
            invalid.append(f"{key}: {exc}")
            continue
        if pred.shape != ref.shape:
            invalid.append(f"{key}: shape mismatch pred={pred.shape} ref={ref.shape}")
            continue
        metrics = image_metrics(pred, ref)
        records.append(Record(name=key, patient=patient_of.get(key, key), metrics=metrics))
        hashes[key] = {"pred_sha256": sha256_file(preds[key]), "ref_sha256": sha256_file(refs[key])}

    if not records:
        raise SystemExit("no valid mask pairs were evaluated")

    macro = macro_summary(records)
    micro = micro_summary(records)
    patient = patient_summary(records)
    boot_macro = bootstrap_macro(records, args.bootstrap, args.seed)
    boot_patient = bootstrap_patient(records, args.bootstrap, args.seed)

    # Per-image CSV
    with open(args.out_dir / "metrics_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["filename", "patient_id", "tp", "fp", "fn", "tn", "dice", "iou", "precision", "recall",
                      "specificity", "hd95_concat_px", "hd95_maxdir_px", "assd_px", "ref_area_px", "pred_area_px"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "filename": r.name, "patient_id": r.patient,
                **{k: ("" if isinstance(v, float) and np.isnan(v) else v) for k, v in r.metrics.items()},
            })

    summary = {
        "n_images": len(records),
        "n_patients": patient.get("n_patients", len(records)),
        "n_hd95_defined": int(sum(not np.isnan(r.metrics["hd95_concat_px"]) for r in records)),
        "n_invalid": len(invalid),
        "definitions": {
            "macro_image": "metric per image, then unweighted mean over images",
            "micro_pooled": "TP/FP/FN pooled over all pixels, metric computed once",
            "patient_mean": "metric per image, averaged within patient, then mean over patients",
        },
        "macro_image": macro,
        "macro_image_ci95": {k: list(v) for k, v in boot_macro.items()},
        "micro_pooled": micro,
        "patient_mean": patient,
        "patient_mean_ci95": {k: list(v) for k, v in boot_patient.items()},
        "bootstrap": {"n": args.bootstrap, "seed": args.seed, "method": "percentile"},
        "threshold": args.threshold,
    }
    (args.out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_table(args.out_dir / "summary_table.md", records, macro, micro, patient,
                        {"macro": boot_macro}, {"patient": boot_patient})

    # Provenance
    import PIL
    provenance = {
        "script": {"path": str(Path(__file__).resolve()), "version": SCRIPT_VERSION,
                   "sha256": sha256_file(Path(__file__).resolve())},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {"python": sys.version, "numpy": np.__version__, "pillow": PIL.__version__,
                        "platform": platform.platform()},
        "inputs": hashes,
        "invalid": invalid,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if invalid:
        (args.out_dir / "missing_or_invalid.txt").write_text("\n".join(invalid) + "\n", encoding="utf-8")

    print(f"Evaluated {len(records)} image pairs ({summary['n_patients']} patients); "
          f"{len(invalid)} invalid; HD95 defined for {summary['n_hd95_defined']}.")
    print(f"Macro Dice {macro['dice']:.4f}  IoU {macro['iou']:.4f}  "
          f"precision {macro['precision']:.4f}  recall {macro['recall']:.4f}")
    print(f"Wrote {args.out_dir/'metrics_summary.json'}")
    return 0


BOOTSTRAP_N = 2000
SEED = 20260922

if __name__ == "__main__":
    raise SystemExit(main())
