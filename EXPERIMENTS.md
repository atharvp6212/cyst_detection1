# Reproducible evaluation protocol

1. Record dataset source, licence, inclusion criteria and patient identifiers.
2. Split by patient before preprocessing; never place images from one patient
   in both training and testing sets.
3. Tune models and thresholds on training/validation data only.
4. Compare a classical baseline, a small CNN and transfer learning using the
   same untouched test set.
5. Report precision, recall, F1, sensitivity, specificity, ROC-AUC, calibration,
   inference latency and memory. Include confidence intervals when possible.
6. Perform an ablation study for normalization, augmentation and segmentation.
7. Analyse false positives and false negatives without exposing patient data.

`evaluate_model.py` writes machine-readable JSON. Its exploratory threshold is
diagnostic only when computed on test data and must not be reported as an
unbiased operating threshold.
