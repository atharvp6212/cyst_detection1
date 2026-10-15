"""Reusable image preprocessing, inference and contour-visualization pipeline."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    contour_count: int
    annotated_image: np.ndarray


def preprocess_for_model(image: np.ndarray, size=(128, 128)) -> np.ndarray:
    if image is None or image.size == 0:
        raise InvalidImageError('The uploaded file is not a readable image.')
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.astype('float32') / 255.0, axis=0)


def analyse_image(image_path: str | Path, model, threshold: float = 0.5) -> DetectionResult:
    image = cv2.imread(str(image_path))
    model_input = preprocess_for_model(image)
    probability = float(np.asarray(model.predict(model_input, verbose=0)).reshape(-1)[0])
    probability = min(1.0, max(0.0, probability))
    infected = probability >= threshold

    annotated = image.copy()
    contour_count = 0
    if infected:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meaningful = [contour for contour in contours if cv2.contourArea(contour) >= 25]
        contour_count = len(meaningful)
        cv2.drawContours(annotated, meaningful, -1, (0, 255, 0), 2)

    confidence = probability if infected else 1.0 - probability
    return DetectionResult(
        label='Infected' if infected else 'Not Infected', confidence=confidence,
        contour_count=contour_count, annotated_image=annotated,
    )
