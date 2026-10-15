"""Evaluate the saved classifier on class-directory test data."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from detection.services import preprocess_for_model
from detection.evaluation import best_f1_threshold, expected_calibration_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cyst_detection_model.h5')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('evaluation_results.json'))
    args = parser.parse_args()
    model = load_model(args.model)
    labels, probabilities = [], []
    for label, class_name in enumerate(('not_infected', 'infected')):
        for path in sorted((args.data / class_name).glob('*')):
            image = cv2.imread(str(path))
            if image is None:
                continue
            probability = float(np.asarray(model.predict(preprocess_for_model(image), verbose=0)).reshape(-1)[0])
            labels.append(label)
            probabilities.append(probability)
    if not labels:
        raise SystemExit('No readable evaluation images were found.')
    predictions = [int(value >= 0.5) for value in probabilities]
    report = classification_report(
        labels, predictions, target_names=['not_infected', 'infected'], output_dict=True
    )
    matrix = confusion_matrix(labels, predictions).tolist()
    best_threshold, validation_f1 = best_f1_threshold(labels, probabilities)
    results = {
        'sample_count': len(labels), 'threshold': 0.5,
        'classification_report': report, 'confusion_matrix': matrix,
        'expected_calibration_error': expected_calibration_error(labels, probabilities),
        'exploratory_best_f1_threshold': best_threshold,
        'exploratory_best_f1': validation_f1,
    }
    if len(set(labels)) == 2:
        results['roc_auc'] = roc_auc_score(labels, probabilities)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
