"""Framework-independent metrics for reproducible binary-model evaluation."""

import random

import numpy as np


def expected_calibration_error(labels, probabilities, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.size == 0 or bins <= 0:
        raise ValueError('Use equal non-empty arrays and a positive bin count.')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must be between zero and one.')
    predicted = probabilities >= 0.5
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        if np.any(mask):
            accuracy = np.mean(predicted[mask] == labels[mask])
            confidence = np.mean(np.maximum(probabilities[mask], 1 - probabilities[mask]))
            error += np.mean(mask) * abs(accuracy - confidence)
    return float(error)


def best_f1_threshold(labels, probabilities, candidates=None) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.asarray(candidates if candidates is not None else np.linspace(0.05, 0.95, 19))
    best = (0.5, -1.0)
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = np.sum((predicted == 1) & (labels == 1))
        fp = np.sum((predicted == 1) & (labels == 0))
        fn = np.sum((predicted == 0) & (labels == 1))
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def patient_level_split(records, test_fraction=0.2, seed=42):
    """Split records by patient ID to prevent image-level leakage."""
    if not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be between zero and one.')
    patient_ids = sorted({record['patient_id'] for record in records})
    random.Random(seed).shuffle(patient_ids)
    count = max(1, round(len(patient_ids) * test_fraction))
    test_ids = set(patient_ids[:count])
    return (
        [record for record in records if record['patient_id'] not in test_ids],
        [record for record in records if record['patient_id'] in test_ids],
    )
