# Lesion-Guided Multimodal Learning for WLI–NBI Endoscopy

Code associated with the manuscript:

**Registration-Free Lesion-Guided Fusion of White-Light and Narrow-Band Endoscopy for Differentiating Gastric Precancerous Lesions from Early Gastric Cancer: A Retrospective Diagnostic Imaging Study**

This repository implements the two-stage pipeline described in the manuscript:

1. WLI lesion segmentation using PSA-CARAFE-ResUNet.
2. Registration-free WLI–NBI classification using a lesion-guided seven-channel WLI branch, an independent NBI branch, and cross-guided attention (CGA) fusion.

## Release

**Recommended manuscript release:** `v1.0.1-final`

The release includes training, evaluation, patient-level splitting, dataset auditing, reader-study analysis, benchmarking, and manuscript statistical utilities.

## Important experiment naming

The manuscript uses the following terminology:

- **Lesion-guided WLI only:** raw WLI + predicted lesion mask + mask-weighted WLI, single WLI branch.
- **NBI only:** NBI branch only.
- **Unguided WLI+NBI:** raw WLI + NBI without lesion-mask guidance.
- **Predicted-mask WLI+NBI:** fully automated inference using the segmentation model's predicted WLI lesion mask.
- **Labeled-mask WLI+NBI:** manual consensus reference mask; used as an ideal-localization reference, not as the deployed workflow.

The final manuscript result of AUC 0.9534 corresponds to the **predicted-mask WLI+NBI** configuration.

## Model-selection protocol

- Patient-level training/development/test separation is used.
- Data augmentation is restricted to the training set.
- The classifier checkpoint is selected by **development-set patient-level AUC**.
- The operating threshold is selected **only on the development set** by maximizing the Youden index.
- The selected model parameters and threshold are frozen before held-out test evaluation.

## Manuscript statistics

`manuscript_statistics.py` implements:

- Hanley–McNeil asymptotic 95% confidence intervals for AUC.
- Paired DeLong tests for correlated AUCs.

The script expects patient-level prediction CSV files with:

```text
patient_id,label,probability
```

The reader-study utility supports 1–5 confidence scores and constructs the signed ordinal score used for reader AUC in the manuscript.

## Data availability

The original patient-level endoscopic images and clinical data are **not included** in this repository.

The study ethics approval permits publication of the research findings and de-identified representative images used in the article. However, institutional ethics, patient-privacy, and data-governance restrictions do not permit distribution of the complete original patient-level clinical image dataset to external parties.

Accordingly, this repository provides the implementation code but not the clinical dataset.

## Environment

The manuscript implementation used:

- Python 3.12
- PyTorch 2.6.0
- CUDA 12.4
- NVIDIA GeForce RTX 4090

See `requirements.txt` and `requirements-tested.txt` for dependencies.

## Typical workflow

```bash
python audit_dataset.py --help
python split_dataset.py --help
python train_segmentation.py --help
python evaluate_segmentation.py --help
python train_classifier.py --help
python evaluate_classifier.py --help
python manuscript_statistics.py --help
python analyze_reader_study.py --help
```

See `docs/` for data format, experiments, and reproducibility notes.

## Reproducibility scope

The repository provides the source code and analysis utilities associated with the manuscript. Because the original clinical image dataset cannot be externally distributed, third parties cannot reproduce the reported numerical results from this repository alone without an appropriately governed dataset.

## License and third-party notices

See `LICENSE`, `THIRD_PARTY_NOTICES.md`, and `licenses/`.

## Citation

If this code is useful, please cite the associated manuscript after publication. Until then, please cite the GitHub repository:

Cai H. *Lesion-Guided-Multimodal-Learning*. GitHub.  
https://github.com/luenCh1/Lesion-Guided-Multimodal-Learning
