# Dataset partition used for training and evaluation

This file records the partition that was used to train and evaluate the models
reported in the manuscript.

## Cohort and splitting

A total of 1,220 patients were included after screening. Patients were
stratified by diagnostic class and divided into disjoint training, development
and held-out test partitions at an 8:1:1 ratio. Every patient contributed one
paired WLI–NBI image set, and `sample_id` values are unique per patient, so no
patient appears in more than one partition.

## Partition sizes

| Partition | Patients | EGC (study-defined) | GPL | Used for |
|---|---|---|---|---|
| Training | 976 | 464 | 512 | network fitting |
| Development | 122 | 58 | 64 | configuration and stopping decisions |
| Held-out test | 122 | 58 | 64 | single evaluation, reported results |

The study-defined EGC class is the composite positive class (invasive early
adenocarcinoma plus high-grade intraepithelial neoplasia). The GPL class is the
composite negative class (chronic atrophic gastritis, intestinal metaplasia,
low-grade and moderate-grade intraepithelial neoplasia).

## Construction

1. A manifest CSV lists every image with columns
   `sample_id, label, wli_path, nbi_path, mask_path`.
2. The manifest is split with stratification on `label` and a fixed random seed
   (3047) into the three partitions above.
3. Augmentation is applied to the training partition only. Development and test
   partitions are evaluated without augmentation.
4. Class frequencies differ between partitions; the classifier uses
   class-weighted cross-entropy computed from the training partition.

## Reproducing the partition

```bash
python split_dataset.py --manifest all.csv --out-dir splits/ --seed 3047
```

The script writes one manifest per partition. The resulting composition depends
on the supplied manifest; the counts in the table above correspond to the
manifest used for the reported runs.

## Note on image availability

The image data are not redistributable because of institutional restrictions.
The manifest is therefore described here by schema and by partition sizes, not
by `sample_id` values.
