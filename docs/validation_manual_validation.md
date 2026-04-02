# Validation Manual Validation

Automated tests are necessary for the validation layer, but they are not enough on their own.

The tests prove that the validation rules behave correctly on controlled examples. You should still run one manual end-to-end check on the real manifest before relying on downstream training.

## What The Validation Layer Is Supposed To Do

The validation layer reads the manifest created by ingestion and checks that it is safe enough to pass to the next stage.

For the current simple version, it checks:

- the manifest file exists
- the required columns exist
- required values are not empty
- every image path exists
- every label name is valid
- every label ID matches the configured mapping
- every `sample_id` is unique
- class counts can be summarized

## How To Run It Manually

From the project root, install dependencies first if needed:

```bash
pip install -r requirements.txt
```

Run validation using the config file:

```bash
python scripts/run_validation.py
```

Override the manifest path or report path if needed:

```bash
python scripts/run_validation.py \
  --manifest-path data/processed/manifest.csv \
  --output-report data/reports/validation_report.json \
  --approved-manifest data/processed/validated_manifest.csv
```

## What Files To Expect

After a successful run, you should see:

- `data/reports/validation_report.json`
- `data/processed/validated_manifest.csv`

If validation fails, the report should still be created, but the validated manifest should not be promoted.

## How To Inspect The Result

Check that the report exists:

```bash
ls data/reports/validation_report.json
```

Print the report:

```bash
cat data/reports/validation_report.json
```

If validation passed, inspect the validated manifest:

```bash
head -n 5 data/processed/validated_manifest.csv
```

## Acceptance Criteria

Treat validation as successful only if all of these are true:

1. The command exits without an exception.
2. `data/reports/validation_report.json` is created.
3. The report contains `passed: true`.
4. The report contains `total_rows` greater than `0`.
5. The report contains a non-empty `class_distribution`.
6. No errors are listed in the report.
7. `data/processed/validated_manifest.csv` is created.
8. The validated manifest has the same required columns as the original manifest.

## Failure Criteria

Treat validation as failed if any of these happen:

- the manifest file is missing
- required columns are missing
- a row has an empty required field
- an image path does not exist
- a label name is not recognized
- a label ID does not match the configured mapping
- a `sample_id` appears more than once

## Suggested Manual Spot Checks

Even if the report says validation passed, do these quick checks once:

```bash
python - <<'PY'
import csv
import json

with open('data/reports/validation_report.json', encoding='utf-8') as handle:
    report = json.load(handle)

print('passed:', report['passed'])
print('rows:', report['total_rows'])
print('class_distribution:', report['class_distribution'])

with open('data/processed/validated_manifest.csv', newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

print('validated rows:', len(rows))
print('columns:', rows[0].keys() if rows else [])
PY
```

If the report and the validated manifest both look correct, the validation layer is behaving as expected for the current simple design.
