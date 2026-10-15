# Cyst Image Classification and Localization Prototype

An academic computer-vision prototype that combines a saved neural-network
classifier with OpenCV preprocessing and contour visualization. It is intended
for experimentation and education, not clinical diagnosis.

## Improvements in this version

- Testable preprocessing and inference service
- Lazy model loading and input validation
- Confidence reporting and contour-area filtering
- Reproducible evaluation script for precision, recall, F1, confusion matrix and ROC-AUC
- Unit tests that do not require loading the full model

## Evaluation

Place held-out images under `not_infected/` and `infected/`, then run:

```bash
python evaluate_model.py --data /path/to/test-data
```

The repository intentionally does not publish accuracy without a documented,
held-out dataset. Future work should add training code, dataset provenance,
patient-independent splitting where applicable, model comparisons and Grad-CAM.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-modern.txt
python manage.py migrate
python manage.py test
python manage.py runserver
```
