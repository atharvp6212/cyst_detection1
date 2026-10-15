from tempfile import NamedTemporaryFile
from unittest.mock import Mock

import cv2
import numpy as np
from django.test import SimpleTestCase

from .services import InvalidImageError, analyse_image, preprocess_for_model


class ImagePipelineTests(SimpleTestCase):
    def test_preprocessing_normalizes_and_resizes(self):
        batch = preprocess_for_model(np.full((20, 30, 3), 255, dtype=np.uint8))
        self.assertEqual(batch.shape, (1, 128, 128, 3))
        self.assertAlmostEqual(float(batch.max()), 1.0)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(InvalidImageError):
            preprocess_for_model(None)

    def test_inference_returns_bounded_confidence(self):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        with NamedTemporaryFile(suffix='.jpg') as handle:
            cv2.imwrite(handle.name, image)
            model = Mock()
            model.predict.return_value = np.array([[0.8]])
            result = analyse_image(handle.name, model)
        self.assertEqual(result.label, 'Infected')
        self.assertAlmostEqual(result.confidence, 0.8)
