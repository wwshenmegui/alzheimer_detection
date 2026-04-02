# Evaluation Layer Manual Validation

Automated tests are necessary for the evaluation layer, but you should still run one manual end-to-end check on the real feature artifact and trained model before relying on the reported metrics.

## What The Evaluation Layer Does

The current evaluation layer is intentionally simple.

It does only this:

- reads `data/processed/features.npz`
- reads `models/trained/logistic_regression.pkl`
- uses the precomputed `test` split from the feature artifact
- runs predictions on the test split
- computes accuracy, macro precision, macro recall, macro F1, and a confusion matrix
- saves a JSON evaluation report

## How To Run It Manually

From the project root:

```bash
pip install -r requirements.txt
python scripts/run_evaluation.py
```

That uses the paths defined in `configs/training.yaml` and the precomputed split labels stored in the feature artifact.
It also prints progress logs to the terminal.

If you want to override them:

```bash
python scripts/run_evaluation.py \
  --input-features data/processed/features.npz \
  --input-model models/trained/logistic_regression.pkl \
  --output-report data/reports/evaluation_report.json \
  --log-level INFO
```

## Expected Output Files

After a successful run, you should see:

- `data/reports/evaluation_report.json`

## How To Inspect The Result

Check that the report exists:

```bash
ls data/reports/evaluation_report.json
```

Print the report:

```bash
cat data/reports/evaluation_report.json
```

Inspect the key metrics quickly:

```bash
python - <<'PY'
import json

with open('data/reports/evaluation_report.json', encoding='utf-8') as handle:
    report = json.load(handle)

print('passed:', report['passed'])
print('accuracy:', report['accuracy'])
print('precision_macro:', report['precision_macro'])
print('recall_macro:', report['recall_macro'])
print('f1_macro:', report['f1_macro'])
print('class_ids:', report['class_ids'])
print('confusion_matrix size:', len(report['confusion_matrix']), 'x', len(report['confusion_matrix'][0]))
PY
```

## Acceptance Criteria

Treat the evaluation layer as successful only if all of these are true:

1. The command exits without an exception.
2. `data/reports/evaluation_report.json` is created.
3. The report contains `passed: true`.
4. The report contains a positive `test_rows` value.
5. The report contains `accuracy`, `precision_macro`, `recall_macro`, and `f1_macro`.
6. The report contains a square confusion matrix whose dimensions match the number of classes.
7. The report contains the expected class IDs.

## Failure Criteria

Treat the evaluation layer as failed if any of these happen:

- the feature artifact is missing
- the trained model file is missing
- the feature artifact is empty or malformed
- the validation split recreation fails
- the output report file is missing after a reported success

If the report exists and the metrics structure looks correct, the current evaluation layer is working as intended.
