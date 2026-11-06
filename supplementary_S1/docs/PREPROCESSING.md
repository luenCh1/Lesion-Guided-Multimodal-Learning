# Preprocessing and training configuration

These are the settings used for the reported runs. They match Section 2 of the
manuscript; if the two ever disagree, the manuscript is authoritative.

## Input construction

| Step | Setting |
|---|---|
| White-light (WLI) frame | read as RGB, resized to 256 × 256 |
| Narrow-band (NBI) frame | read as RGB, resized to 256 × 256 |
| Lesion mask | read as 8-bit grayscale, binarised, resized with nearest-neighbour interpolation |
| Mask-weighted WLI | WLI with pixels outside the lesion mask suppressed |
| WLI branch input | 7 channels = original WLI (3) + binary lesion mask (1) + mask-weighted WLI (3) |
| NBI branch input | 3 channels (RGB) |
| Normalisation | per channel, ImageNet statistics: mean (0.485, 0.456, 0.406), SD (0.229, 0.224, 0.225) |
| Feature resolution | 192 channels at 16 × 16 after each backbone |

## Augmentation (training partition only)

| Target | Transform |
|---|---|
| WLI image + lesion mask | synchronised random rotation within ±15°, horizontal flip p = 0.5 |
| NBI image | independently sampled rotation within ±15°, horizontal flip p = 0.5 |
| Segmentation input | rotation ±35° (p = 0.3), horizontal flip (p = 0.3), vertical flip (p = 0.3) |

The two classification branches are augmented independently so that no artificial
pixel-level correspondence is imposed between WLI and NBI. No colour jitter and no
cutout are used.

## Classifier training

| Item | Value |
|---|---|
| Backbone | two independent RepVGG-A0 branches |
| Fusion | cross-guided attention (channel + spatial attention, sigmoid pixel gate) |
| Refinement | 1 × 1 convolution + two bottleneck residual blocks, then global pooling |
| Head | binary classification |
| Optimiser | AdamW |
| Initial learning rate | 1 × 10⁻³ |
| Weight decay | 1 × 10⁻⁴ |
| Batch size | 8 |
| Epochs | up to 100 |
| LR schedule | cosine annealing, minimum 1 × 10⁻⁶ |
| Loss | inverse-frequency class-weighted cross-entropy |
| Training masks | manual consensus masks |
| Masks at development / test | predicted masks from the segmentation network |
| Checkpoint selection | highest development-set AUC |
| Decision threshold | 0.5 |
| Parameters | ≈ 22.39 M |

## Segmentation network

| Item | Value |
|---|---|
| Architecture | PSA-CARAFE-ResUNet (residual encoder–decoder, polarized self-attention, CARAFE upsampling) |
| Input | WLI only, 256 × 256, ImageNet normalisation |
| Optimiser | AdamW |
| Initial learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Batch size | 16 |
| Epochs | up to 100 |
| LR schedule | ReduceLROnPlateau on development-set Dice (factor 0.5, patience 15) |
| Loss | unweighted sum of binary cross-entropy and soft Dice |
| Binarisation | 0.5 |
| Random seed | 3047 |
| Parameters | ≈ 20.56 M |

## Software and hardware

| Item | Value |
|---|---|
| Python | 3.12 |
| PyTorch | 2.6.0 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 4090 |

## Random seeds

Python, NumPy and PyTorch seeds were fixed for each run. Bitwise reproducibility
across runs is not guaranteed because cuDNN may select different convolution
algorithms; statistical reproducibility (same mean and spread) is expected.
