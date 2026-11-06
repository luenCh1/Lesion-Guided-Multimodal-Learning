"""Inference: load a trained checkpoint and export per-image probabilities.

Usage
-----
python infer.py --ckpt path/to/model.pt --manifest manifest.csv --out preds.csv

The checkpoint is expected to contain:
    {"model_state": {...}, "kwargs": {...}, ...}

Manifest CSV columns (paths relative to --data-root):
    sample_id, label, wli_path, nbi_path, mask_path
    label: EGC / GPL

No patient identifiers or cohort statistics are embedded in this script; all
counts are derived at runtime from the manifest.
"""
import argparse
import csv
import csv as _csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_pipeline import MultimodalDataset          # noqa: E402
from models.classification import LesionGuidedDualModel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True, help="root directory for image paths in the manifest")
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mask-source", choices=["manual", "predicted"], default="manual")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model_state") or ck.get("state_dict")
    if sd is None:
        raise SystemExit("checkpoint does not contain model_state")
    kw = ck.get("kwargs") or dict(modality_mode="dual", wli_input_mode="full", use_cga=True)

    model = LesionGuidedDualModel(pretrained=False, **kw).to(dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing:
        raise SystemExit("state_dict mismatch, missing keys: %s" % missing[:5])
    model.eval()

    ds = MultimodalDataset(a.manifest, a.data_root, size=(256, 256), training=False,
                           mask_source=a.mask_source)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    rows = list(_csv.DictReader(open(a.manifest, encoding="utf-8")))

    probs = []
    with torch.no_grad():
        for b in loader:
            logits = model(b["nbi"].to(dev), b["wli"].to(dev), b["mask"].to(dev))
            probs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "y", "p"])
        for r, pi in zip(rows, p):
            w.writerow([r["sample_id"], 1 if r["label"] == "EGC" else 0, "%.6f" % float(pi)])
    print("wrote %d rows -> %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
