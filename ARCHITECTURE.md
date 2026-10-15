# Architecture

The web layer validates uploads and delegates to a reusable inference service.
Evaluation utilities remain independent of TensorFlow so splitting, threshold
selection and calibration can be tested quickly. The command-line evaluator
loads the saved model, processes a fixed class-directory dataset and emits JSON
for later plotting or statistical comparison.
