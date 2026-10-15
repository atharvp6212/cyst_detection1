"""Evaluate the saved classifier on class-directory test data."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from detection.services import preprocess_for_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cyst_detection_model.h5')
    parser.add_argument('--data', required=True, type=Path)
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
    print(classification_report(labels, predictions, target_names=['not_infected', 'infected']))
    print('Confusion matrix:\n', confusion_matrix(labels, predictions))
    if len(set(labels)) == 2:
        print('ROC-AUC:', roc_auc_score(labels, probabilities))


if __name__ == '__main__':
    main()
