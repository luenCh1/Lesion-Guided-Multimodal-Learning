# Training and evaluation protocol

This document matches Section 2 of the manuscript.

## Partitions

Three disjoint patient-level partitions are used: training, development and
held-out test. Sizes and class composition are given in `SPLIT.md`. The
development partition is used for checkpoint selection; the test partition is
evaluated once.

## Classification

1. Train with manual consensus lesion masks on the training partition; evaluate
   on the development partition with predicted masks from the segmentation
   network.
2. After each epoch, compute the development-set AUC and retain the checkpoint
   with the highest value.
3. Evaluate the retained checkpoint once on the held-out test partition, using
   predicted masks.
4. All threshold-dependent metrics use a fixed threshold of 0.5. The threshold
   is not tuned on any partition.

The training script exposes this behaviour through its arguments:

```bash
python train_classifier.py \
  --train-csv splits/train.csv --val-csv splits/dev.csv \
  --train-mask-source manual --val-mask-source predicted \
  --data-root /path/to/images --epochs 100 --batch-size 8
```

## Segmentation

Trained on the training partition; ReduceLROnPlateau is driven by development-set
Dice. Predictions are binarised at 0.5 and used both for the segmentation metrics
and as the mask source for classifier development and test evaluation.

## Metrics

### Classification

- Image-level (equivalently patient-level, one image pair per patient) AUC, using
  the Mann–Whitney formulation so that the value is threshold-free.
- Accuracy, sensitivity, specificity, PPV, NPV and F1 at the fixed threshold.
- Proportion intervals: Wilson score method.
- AUC intervals: percentile bootstrap over patients (2,000 resamples).
- Correlated AUCs compared with DeLong tests; nominal P values, comparisons
  exploratory.

### Segmentation

- Dice, IoU, precision and recall computed per image from the same pixel counts
  and averaged over the test images.
- HD95: the mask surface is the mask minus a one-pixel, four-connected erosion;
  Euclidean distances are computed in both directions on the 256 × 256 grid with
  unit pixel spacing, pooled, and the 95th percentile taken. Pairs for which
  either surface is undefined contribute no value.
- Intervals: percentile bootstrap, 2,000 resamples.
