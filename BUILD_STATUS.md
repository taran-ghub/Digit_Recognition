# Build verification — 19 October 2026

## Delivered

Source for dataset preparation, both CNNs, training, threshold calibration,
segmentation, inference, Streamlit upload UI, held-out evaluation, example images,
regression tests and Windows setup instructions.

## Executed successfully in the build environment

- Installed pinned requirements on Linux / Python3.12 (Windows/Python3.11 instructions
  supplied; Windows itself was not available for testing).
- Six pytest tests passed: blank input, polarity/geometry including a thin digit,
  leading zeros and whole-string rejection, decimal mark retention, transparent
  background, and edit distance.
- Both CNNs completed a training batch, saved to .keras and reloaded with correct
  output shapes.
- A short end-to-end training smoke run completed for both models:1000 MNIST
  training digits, generated negatives only,2 epochs; separate200 validation
  examples. Calibration produced thresholds. This run lacked EMNIST and is NOT
  the project's final experiment.
- Evaluation ran on a200-example test fixture and10 synthetic strings, exercising
  metrics export. These smoke metrics are deliberately not presented as results.
- Streamlit started without exceptions in its missing-model state.
- Streamlit upload paths executed without exceptions for a synthetic007 image
  (component visualization and score table) and a blank image (rejection).
- Pinned PyArrow18.1.0 after detecting an incompatibility between the latest
  transitive PyArrow release and TensorFlow's NumPy1.26 requirement.

## Training/data blocker

MNIST downloaded successfully. The first generic dataset loader attempted an
unnecessary cloud-metadata check and was blocked. That dependency was removed;
no cloud credentials or metadata are required by the delivered source.

The replacement direct official NIST EMNIST download returned HTTP403 in this
environment. A --emnist-zip option supports an official archive downloaded on
another computer, with the same SHA256 integrity check. Full preparation on
EMNIST and full training/evaluation therefore remain unverified here.

No pretrained weights, final accuracy claims or invented local-volunteer data are
included. Empty models/ is intentional. The interface shows training instructions
until digit.keras, validator.keras and thresholds.json exist together.

## Before a class presentation

1. Download/prepare the official data.
2. Run full training and evaluation (README commands).
3. Collect independent, labelled handwriting and non-digit test inputs.
4. Review false accepts, false rejects and segmentation mistakes.
5. Put actual evaluation results and limitations into the report.

This version supports image uploads; drawing inside the app is not yet included.
