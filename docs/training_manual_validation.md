# Training Layer Manual Validation

Automated tests are necessary for the training layer, but you should still run one manual end-to-end check on the real feature artifact before using the model downstream.

## What The Training Layer Does

The current training layer is intentionally simple.

It does only this:

- reads `data/processed/features.npz`
- flattens grayscale image tensors into 2D feature vectors
- uses the precomputed `train` and `validation` splits from the feature artifact
- trains a multinomial logistic regression baseline
- evaluates validation accuracy
- saves the trained model and a JSON training report

## How To Run It Manually

From the project root:

```bash
pip install -r requirements.txt
python scripts/run_training.py
```

That uses the paths and hyperparameters defined in `configs/training.yaml`.
It also prints progress logs to the terminal.

If you want to override them:

```bash
python scripts/run_training.py \
  --input-features data/processed/features.npz \
  --output-model models/trained/logistic_regression.pkl \
  --output-report data/reports/training_report.json \
  --max-iter 500 \
  --log-level INFO
```

## Expected Output Files

After a successful run, you should see:

- `models/trained/logistic_regression.pkl`
- `data/reports/training_report.json`

## How To Inspect The Result

Check that the files exist:

```bash
ls models/trained/logistic_regression.pkl
ls data/reports/training_report.json
```

Print the report:

```bash
cat data/reports/training_report.json
```

Inspect the saved model quickly:

```bash
python - <<'PY'
import pickle

with open('models/trained/logistic_regression.pkl', 'rb') as handle:
    model = pickle.load(handle)

print('classes:', model.classes_)
print('coef shape:', model.coef_.shape)
PY
```

## Acceptance Criteria

Treat the training layer as successful only if all of these are true:

1. The command exits without an exception.
2. `models/trained/logistic_regression.pkl` is created.
3. `data/reports/training_report.json` is created.
4. The report contains `passed: true`.
5. The report contains positive `train_rows` and `validation_rows` values.
6. The report contains a non-negative `validation_accuracy`.
7. The saved model can be loaded with Python `pickle`.
8. The model contains the expected class IDs.

## Failure Criteria

Treat the training layer as failed if any of these happen:

- the feature artifact is missing
- the feature artifact is empty or malformed
- the train/validation split fails
- model training raises an exception
- the output model file is missing after a reported success
- the training report is missing after a reported success

If the report and the saved model both look correct, the current training layer is working as intended.
