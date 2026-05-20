# FairFace-ID: Auditing and Mitigating Demographic Bias in Multiracial Face Recognition

This repository contains a course-project implementation for auditing demographic performance gaps in multiracial face verification systems and testing a simple mitigation method based on threshold calibration.

The project is designed to be safe, reproducible, and GitHub-ready:

- It **does not include face images or pretrained model weights**.
- It works immediately with a **synthetic demo** so the pipeline can be tested without downloading biometric data.
- It can also run on real face-verification datasets such as RFW or a course-provided dataset when the user has lawful access and consent/permission to use the data.

## Project idea

Face recognition models often report high overall accuracy, but the same global threshold can produce different false positive and false negative rates across demographic groups. This project evaluates those gaps using group-wise metrics and compares:

1. A baseline global threshold.
2. Group-aware threshold calibration on a validation split.

The aim is not to deploy a surveillance system. The aim is to build an academic auditing tool that helps identify and reduce unfair error distributions.

## Repository structure

```text
fairface-id-bias-audit/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── scripts/
│   ├── run_demo.py
│   └── run_pipeline.py
├── src/fairface_id/
│   ├── __init__.py
│   ├── data.py
│   ├── embeddings.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── simulate.py
│   └── thresholding.py
└── tests/
    └── test_metrics.py
```

## Quick start: run the synthetic demo

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_demo.py --out_dir outputs/demo
```

Expected outputs:

```text
outputs/demo/
├── synthetic_pairs.csv
├── global_per_group_metrics.csv
├── calibrated_per_group_metrics.csv
├── global_fairness_summary.csv
├── calibrated_fairness_summary.csv
└── summary.json
```

## Run on your own face-verification score file

If you already have a CSV containing image-pair scores, use:

```bash
python scripts/run_pipeline.py \
  --scores_csv data/pairs_with_scores.csv \
  --out_dir outputs/real_eval \
  --group_col group \
  --label_col label \
  --score_col score
```

The input CSV should contain:

| column | meaning |
|---|---|
| `label` | 1 for same identity, 0 for different identity |
| `score` | face similarity score, usually cosine similarity |
| `group` | demographic group used for fairness auditing, such as African, Asian, Caucasian, Indian, etc. |

## Optional: extract embeddings from images

The repository includes an optional helper in `src/fairface_id/embeddings.py` for extracting face embeddings with `facenet-pytorch`. Install the optional image dependencies first:

```bash
pip install -r requirements-image.txt
```

Then extract embeddings:

```python
from fairface_id.embeddings import extract_embeddings_from_paths

extract_embeddings_from_paths(
    image_paths=["data/images/person1_001.jpg", "data/images/person2_001.jpg"],
    output_npz="outputs/embeddings.npz",
    device="auto"
)
```

For a real dataset, prepare:

1. `metadata.csv` with image paths and group labels.
2. `pairs.csv` with `path1`, `path2`, `label`, and `group`.
3. Embeddings saved as `.npz`.
4. Scores joined into a CSV using cosine similarity.

## Ethical and legal notes

- Use only datasets that your team is allowed to access and analyze.
- Do not scrape personal images or identify people without consent.
- Keep raw biometric images out of the GitHub repository.
- Report group-level statistics rather than publishing individual predictions.
- Treat race labels as socially constructed annotation categories, not biological ground truth.

## Suggested datasets for an academic report

- RFW: Racial Faces in-the-Wild, for race-balanced verification testing.
- FairFace: face attribute dataset balanced across race, gender, and age categories.
- Course-provided or competition-provided datasets, if the license permits academic use.

## Main metrics

The code reports:

- Accuracy
- Precision
- Recall / True Positive Rate
- Specificity / True Negative Rate
- False Positive Rate
- False Negative Rate
- F1-score
- Balanced accuracy
- ROC-AUC, when `scikit-learn` is installed
- Group fairness gaps, such as max-min TPR gap, max-min FPR gap, and max-min accuracy gap

## Citation note

This repository is for a course proposal and prototype. If you use public datasets or pretrained models, cite the corresponding papers and dataset licenses in your final report.
