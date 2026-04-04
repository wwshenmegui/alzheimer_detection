# Alzheimer Detection

## What This Project Is About

Alzheimer Detection is a simple end-to-end machine learning project for classifying MRI brain images into Alzheimer's disease stages.

The current version includes:

- a training pipeline
- a simple grayscale image preprocessing flow
- a baseline logistic regression model
- an evaluation pipeline
- a small FastAPI serving backend for prediction

The current dataset source is based on the Kaggle Alzheimer's multiclass MRI dataset used in the training pipeline.

## Project Structure

```text
alzheimer_detection/
├── configs/
│   └── training.yaml
├── data/
│   ├── processed/
│   ├── raw/
│   └── reports/
├── docs/
│   ├── evaluation_manual_validation.md
│   ├── features_manual_validation.md
│   ├── ingestion_manual_validation.md
│   ├── serving_manual_validation.md
│   ├── training_manual_validation.md
│   └── validation_manual_validation.md
├── models/
│   └── trained/
├── scripts/
│   ├── run_evaluation.py
│   ├── run_features.py
│   ├── run_ingestion.py
│   ├── run_server.py
│   ├── run_training.py
│   └── run_validation.py
├── src/
│   ├── serving/
│   ├── shared/
│   └── training/
├── tests/
│   ├── serving/
│   └── training/
└── requirements.txt
```

Key modules:

- `src/training/ingestion`: download and manifest building
- `src/training/validation`: manifest checks and validation report
- `src/training/features`: grayscale preprocessing, tensor generation, and explicit train/validation/test split creation
- `src/training/models`: baseline model training
- `src/training/evaluation`: offline evaluation on the test split
- `src/serving`: prediction backend
- `src/shared`: preprocessing shared by training and serving

## How To Use

### 1. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

For local quality checks:

```bash
ruff check .
pytest tests/training tests/serving -q
```

### 2. Train The Model

The training pipeline is split into several steps.

#### Step 1: Ingest Data

If you want to download the dataset through KaggleHub:

```bash
python scripts/run_ingestion.py --download
```

If the dataset already exists locally:

```bash
python scripts/run_ingestion.py \
  --dataset-root data/raw/alzheimers_multiclass \
  --output-manifest data/processed/manifest.csv
```

#### Step 2: Validate Manifest

```bash
python scripts/run_validation.py
```

#### Step 3: Build Features

```bash
python scripts/run_features.py
```

This step creates grayscale `H x W x 1` tensors and stores explicit dataset splits with ratio `6:2:2` for:

- train
- validation
- test

#### Step 4: Train Model

```bash
python scripts/run_training.py
```

#### Step 5: Evaluate Model

```bash
python scripts/run_evaluation.py
```

### 3. Run The Server

Start the API server:

```bash
python scripts/run_server.py
```

By default the server runs at:

```text
http://127.0.0.1:8000
```

### 4. Call The Backend

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/your_mri_image.png"
```

Expected response fields:

- `predicted_label`
- `predicted_label_id`
- `probabilities`
- `input_shape`
- `model_name`

## Workflow

The end-to-end workflow is:

```text
raw MRI images
-> ingestion
-> manifest
-> validation
-> validated manifest
-> feature building
-> features.npz with train/validation/test splits
-> model training
-> trained model artifact
-> evaluation
-> evaluation report
-> serving API
-> prediction response
```

More specifically:

1. Ingestion scans the dataset and builds a manifest.
2. Validation checks the manifest structure and file paths.
3. Feature building converts images to grayscale tensors and assigns `6:2:2` dataset splits.
4. Training fits a baseline logistic regression model on the training split and checks validation performance.
5. Evaluation measures model performance on the held-out test split.
6. Serving loads the trained model and exposes prediction through FastAPI.

## Things That Need To Be Improved

The current version is intentionally simple. Important improvements are still needed.

### Model Quality

- replace the baseline logistic regression model with a stronger image model such as a CNN
- improve numerical stability during training on the full dataset
- add model calibration and confidence handling
- compare multiple model families instead of relying on one baseline

### Data Quality And Validation

- ~~add duplicate and near-duplicate checks~~
- ~~add stronger MRI-specific validation rules~~
- ~~add patient-level leakage prevention if future datasets include patient identifiers~~
- ~~add richer dataset statistics and data quality reporting~~

### Evaluation

- add ROC AUC and more detailed per-class analysis
- add confusion matrix plots and saved visual reports
- add cross-validation or external test-set validation
- add fairness and subgroup evaluation if metadata becomes available

### Serving And Productization

- add model versioning and model metadata endpoints
- add authentication and request logging
- add batch inference support
- add structured error handling and monitoring
- ~~add a frontend or clinic-facing UI~~

### Engineering

- split configuration into training and serving configs if the project grows
- ~~add CI automation for tests and linting~~
- add containerization for easier deployment
- add environment setup documentation beyond `requirements.txt`

## CI

GitHub Actions CI is configured in `.github/workflows/ci.yml`.

It currently runs on push to `main` and on pull requests, and performs:

- `ruff check .`
- `pytest tests/training tests/serving -q`

## Notes

- The current preprocessing is grayscale, fixed-size, and shared between training and serving.
- The current dataset split is explicit and stored in the feature artifact, so training and evaluation use the same split assignment.
- The current serving module is intended for local development and testing, not production deployment.