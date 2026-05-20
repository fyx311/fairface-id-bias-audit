# Project Proposal Summary

**Project title:** FairFace-ID: Auditing and Mitigating Demographic Bias in Multiracial Face Recognition

**Goal:** Build a reproducible pipeline that evaluates face-verification performance across demographic groups and compares a baseline global threshold with a group-aware calibrated threshold.

**Task type:** Face verification, not open-world public identification.

**Inputs:**

- `scores.csv` with `label`, `score`, and `group` columns; or
- Raw face images plus `pairs.csv`, then embeddings can be extracted with the optional `facenet-pytorch` helper.

**Outputs:**

- Overall metrics
- Group-wise metrics
- Fairness gap tables
- JSON summary
- Optional charts

**Ethics:** Use only permitted datasets, do not upload raw biometric images to GitHub, and report group-level results rather than individual predictions.
