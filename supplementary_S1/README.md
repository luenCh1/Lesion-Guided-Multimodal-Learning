# Supplementary Model S1 and Code S1

This folder holds the two code components referenced in the manuscript:

- **Supplementary Model S1** — training configuration, preprocessing, partition
  documentation, inference and metric code for the lesion-guided WLI–NBI
  classifier and the PSA-CARAFE segmentation network.
- **Supplementary Code S1** — deterministic evaluator for paired segmentation
  masks.

The network definitions (`models/`), the data pipeline (`data_pipeline.py`), the
training entry point (`train_classifier.py`) and the partitioning utility
(`split_dataset.py`) live at the repository root and are used unchanged.

No image data, patient identifiers or cohort-level statistics are included in this
repository. Sample counts are read at run time from the manifest supplied by the
user.

## Contents

| File | Purpose |
|---|---|
| `infer.py` | load a checkpoint and export per-image probability |
| `metrics_report.py` | AUC, accuracy, sensitivity, specificity, PPV, NPV, F1 with bootstrap CIs |
| `evaluate_segmentation.py` | Dice, IoU, precision, recall, HD95 with bootstrap CIs |
| `docs/PREPROCESSING.md` | input construction, augmentation, optimiser settings |
| `docs/SPLIT.md` | partitions used for the reported runs |
| `docs/PROTOCOL.md` | evaluation protocol and metric definitions |

## Checkpoint format

```python
{
  "model_state": {...},                                  # state_dict
  "kwargs": {"modality_mode": "dual",
             "wli_input_mode": "full",
             "use_cga": True},                           # architecture arguments
  "epoch": 100
}
```

`infer.py` builds the network from `kwargs` before loading `model_state`, so a
checkpoint cannot be silently paired with the wrong architecture.

## Usage

```bash
# inference
python supplementary_S1/infer.py \
    --ckpt checkpoints/model.pt \
    --manifest splits/test.csv \
    --data-root /path/to/images \
    --mask-source predicted \
    --out preds.csv

# classification metrics
python supplementary_S1/metrics_report.py \
    --pred proposed=preds.csv --threshold 0.5 --bootstrap 2000 --out metrics.csv

# segmentation metrics
python supplementary_S1/evaluate_segmentation.py --help
```

## Summary of the reported configuration

| Item | Value |
|---|---|
| Classification input | WLI 7 channels (RGB + mask + mask-weighted WLI), NBI 3 channels, 256 × 256 |
| Classification optimiser | AdamW, lr 1e-3, weight decay 1e-4, batch 8, up to 100 epochs |
| Classification schedule | cosine annealing to 1e-6; checkpoint with highest development-set AUC |
| Segmentation optimiser | AdamW, lr 1e-4, weight decay 1e-5, batch 16, up to 100 epochs |
| Segmentation schedule | ReduceLROnPlateau on development Dice (factor 0.5, patience 15) |
| Decision threshold | 0.5, fixed a priori |
