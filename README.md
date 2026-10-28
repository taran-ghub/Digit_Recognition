# DigitLab — Handwritten Digit and Number Recognition

An implementation of the project proposal: three component-level CNN checks, an
end-to-end CRNN–CTC number reader, validation-based rejection, and format-aware
number recognition. This is a research prototype, not a claim
of perfect non-digit detection. Start with clear uploaded images, not a camera
scene. Draw in Paint and save as PNG if you want to test your own handwriting.

## 1. Windows setup (Python 3.11, 64-bit)

Extract this folder. Open Command Prompt in it (the folder containing app.py).
Run these commands one at a time:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
```

Use a fresh environment; do not install into a different project's environment.
The pins are a reproducible baseline, not a claim that every Windows machine
has been tested. CPU training is supported; no NVIDIA GPU is required. Python
3.11 avoids incompatible wheels on newer Python releases.

## 2. Prepare data, train, evaluate

An internet connection is needed for the first dataset download. EMNIST downloads its full archive (about562 MB) even though we use only letters. Allow several GB of
free disk space and RAM; preparation stores arrays in memory. Training time
varies by computer. Leave the terminal open.

```bat
python -m hdr.prepare
python -m hdr.train
python -m hdr.prepare_character --emnist-zip "C:\Users\YOUR_NAME\Downloads\gzip.zip"
python -m hdr.train_character
python -m hdr.train_sequence
python -m hdr.evaluate
python -m streamlit run app.py
```

The last command prints the local URL for the app. Stop it with Ctrl+C.
After setup, double-click run_windows.bat to reopen it. To inspect the interface
before training, run the Streamlit command: it shows the missing-model state.

If you only want to check that training executes:
```bat
python -m hdr.train --limit 2000 --epochs 1 --models models_smoke
```
This writes to a separate folder and is NOT a final training run. A poor smoke
model may fail the 1% validation false-acceptance constraint; this is intentional.
Run the full default command to create models used by the app.

Linux/macOS activation: `source .venv/bin/activate`; use `python3 -m venv .venv`.

If NIST returns HTTP403 or the automatic download fails, open the official
[NIST EMNIST page](https://www.nist.gov/itl/products-and-services/emnist-dataset)
in your browser and download the Binary/MNIST-format archive (gzip.zip).
Do not extract it. Then run:

```bat
python -m hdr.prepare --emnist-zip "C:\Users\YOUR_NAME\Downloads\gzip.zip"
python -m hdr.train
python -m hdr.evaluate
```

The same official SHA256 is checked for manual and automatic downloads.

## 3. What is implemented

- Shared grayscale preprocessing, automatic polarity, transparent PNG handling,
  aspect-ratio-preserving resizing, and 28x28 centered input.
- A digit CNN: Conv32 -> Pool -> Conv64 -> Pool -> Dense128 -> Dropout -> 10 classes.
- A binary CNN trained on digits versus EMNIST letters and generated negatives.
- A 36-class character CNN trained to distinguish digits 0-9 from case-merged
  letters A-Z before a component can be accepted as a digit.
- An end-to-end CRNN–CTC pathway: CNN feature extraction followed by two
  bidirectional LSTM layers reads a complete handwritten number as a sequence.
  It is trained only on synthetic strings made from the appropriate MNIST split.
- Dual-path acceptance: once the sequence model is trained, the component pathway
  and the full-number sequence pathway must produce the same reading.
- Explicit, selectable post-recognition format rules for a general number, Indian
  mobile number, Indian PIN code, six-digit OTP and eight-digit roll number.
  These are format checks, not identity verification.
- A locally stored optional correction/feedback workflow and an in-app metrics dashboard.
- Live WebRTC camera capture with a five-frame stability check before showing a result.
- Training-only rotation/translation augmentation, Adam, early stopping.
- Validation-only threshold search: maximize digit retention while the original
  non-digit gate and the character gate each meet their configured 5% validation
  false-acceptance targets. This is not a real-world guarantee.
- Projection segmentation for a single row of separated characters. Thin digits
  are retained; detached small marks are flagged rather than silently discarded.
- All three checks must accept every component. Otherwise the whole string is rejected.
- String output preserves `007`. A number is never converted to an integer.
- Streamlit drawing canvas and upload interface with bounding boxes, normalized
  crops, model scores, rejection explanation and downloadable JSON.
- Held-out evaluation and optional independently collected image CSV evaluation.

## 4. How the files connect

| File | Purpose |
|---|---|
| hdr/prepare.py | Downloads datasets, creates separate train/validation/test files |
| hdr/preprocess.py | Makes training/inference inputs consistent; separates characters |
| hdr/models.py | Defines the digit, validator and 36-class character CNNs |
| hdr/train.py | Learns weights and calibrates rejection thresholds |
| hdr/prepare_character.py | Builds the combined digit/letter datasets |
| hdr/train_character.py | Trains and calibrates the 36-class character gate |
| hdr/sequence.py | Creates synthetic number rows and decodes CTC sequences |
| hdr/train_sequence.py | Trains the CNN–BiLSTM–CTC full-number recognizer |
| hdr/validators.py | Transparent number-format rules |
| hdr/inference.py | Loads models and makes accept/reject decisions |
| hdr/evaluate.py | Measures performance without tuning on test data |
| app.py | Upload interface and explanation of the prediction |
| tests/test_pipeline.py | Logic and preprocessing regression tests |

Generated artifacts:
- data/train.npz, val.npz, test.npz, split_ids.npz, manifest.json
- models/digit.keras, validator.keras, character.keras, sequence.keras,
  thresholds.json and training metadata
- reports/evaluation.json: classifier accuracy/F1/recall/confusion matrix,
  validator report, coverage, false accepts/rejects, selective accuracy,
  synthetic exact-string accuracy/CER, latency and individual string errors.

## 5. Data and experimental honesty

MNIST: official 60,000 training images split into 50,000 training and 10,000
validation images, stratified with seed42. Official 10,000 test images stay out
of training and threshold selection. EMNIST Letters: official training partition
split90/10; official test kept separate. Raw EMNIST is transposed to upright; the archive SHA256 and IDX headers are checked.
Both sources pass through exactly the same crop/pad pipeline.

Binary negatives: approximately80% letter images and20% generated plus/equal
symbols, scribbles and noise, balanced against positive digits. Generated splits
use different seeds but share shape families. This is not unseen-family testing;
the report should acknowledge that limitation. Some scribbles can resemble digits.

Synthetic strings use ONLY held-out MNIST test images, with2–6 digits, varied
spacing and vertical offsets. They are a controlled benchmark, not evidence of
performance on actual handwritten numbers. Test crops can recur across synthetic
strings; the strings are not independent handwriting samples.

For a fairer comparison, the output includes confidence-only rejection at the SAME
classifier threshold as the two-model pipeline. This is an ablation, not a fully
independently optimized baseline. No local volunteer data are fabricated.

## 6. Collect real test images

Create data/local/ and save images with separate filenames. Make data/local.csv:

```csv
path,label,writer_id
local/writer01_007.png,007,writer01
local/writer02_A.png,NON_DIGIT,writer02
local/writer03_12A4.png,NON_DIGIT,writer03
```

Read labels as text so leading zeros survive. Use NON_DIGIT for invalid/mixed
inputs, decimals and blanks. For the genuine-number group use only supported
integer strings. Split writers before any local fine-tuning. The CSV evaluator
does not enforce writer separation: you must keep the held-out writers untouched.

```bat
python -m hdr.evaluate --local-csv data/local.csv
```

This adds separate local metrics. Do not tune thresholds on these test results.
Inspect individual errors for splits, merges, omitted or extra characters.
Automatic segmentation ground-truth metrics are not implemented; they need annotated
boxes. Review these separately rather than inventing segmentation accuracy.

## 7. Limits you should explain to your teacher

The component pathway uses projection segmentation, while the sequence pathway
reads the whole row. The final system is still conservative: touching digits can
merge and cause disagreement or rejection; detached strokes can split. Some
difficult segmentation errors can still pass multiple checks. The row checks are
heuristics, not a guaranteed layout detector. Decimals, minus signs, fractions and
multi-line writing are unsupported. Restrict the live demo to1–6 separated digits.

The binary model learns the negatives it sees. It cannot prove arbitrary input is
a digit, and O/0 or I/1 are visually ambiguous. Model scores are not calibrated
probabilities. Accuracy, validation FAR and real-world performance differ.

Changes from the ambitious proposal: upload, an in-app drawing canvas, local feedback,
format rules, dual-path sequence OCR and live-camera capture are implemented. Connected-
component fragment merging, unseen-family negatives, writer study and automatic
segmentation scoring remain future work. Nothing here claims they are done.

## 8. Explain it simply

“First we clean the image and separate its characters. The first neural network
suggests which digit each character is. The second checks whether it resembles a
digit at all. A third model compares it against digits and letters. All checks must
pass thresholds chosen using validation data. A separate sequence model reads the
entire number; it must agree before a number is accepted. We evaluate on separate test data
so we can measure how well it works on examples not used during training.”

Training learns model weights. Validation chooses stopping/thresholds. Testing
measures performance after those decisions are fixed. Rejection improves caution
but can also reject genuine digits: report both error types and coverage.

## Sources

- TensorFlow install: https://www.tensorflow.org/install/pip
- MNIST catalog: https://www.tensorflow.org/datasets/catalog/mnist
- EMNIST catalog: https://www.tensorflow.org/datasets/catalog/emnist
- EMNIST source/orientation: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py

Do not report an accuracy value until reports/evaluation.json exists from a full run.
See BUILD_STATUS.md for what was actually executed when this package was delivered.
