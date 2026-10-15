from django.test import SimpleTestCase

from .evaluation import best_f1_threshold, expected_calibration_error, patient_level_split


class EvaluationTests(SimpleTestCase):
    def test_calibration_error_is_zero_for_certain_correct_predictions(self):
        self.assertEqual(expected_calibration_error([0, 1], [0, 1]), 0)

    def test_selects_threshold_using_validation_labels(self):
        threshold, score = best_f1_threshold([0, 1, 1], [0.1, 0.4, 0.8], [0.3, 0.5])
        self.assertEqual((threshold, score), (0.3, 1.0))

    def test_patient_split_has_no_patient_leakage(self):
        records = [{'patient_id': patient, 'image': index} for patient in 'abcd' for index in range(2)]
        train, test = patient_level_split(records, test_fraction=0.25)
        self.assertTrue({x['patient_id'] for x in train}.isdisjoint({x['patient_id'] for x in test}))
