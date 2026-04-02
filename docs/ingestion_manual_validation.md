# Ingestion Manual Validation

Automated tests are necessary, but they are not sufficient for the ingestion layer.

The automated tests prove that the core code paths work on small synthetic examples. You should still run one manual end-to-end validation whenever one of these is true:

- the ingestion code changes materially
- the dataset source changes
- the config file changes
- you are preparing to hand the project to someone else

## What To Check Manually

The goal of manual validation is to confirm that ingestion works on the real dataset, not just on test fixtures.

You want to verify:

- the dataset can be downloaded or located successfully
- the code finds the correct class folders
- a manifest file is created in the expected location
- manifest rows look correct
- labels are mapped correctly
- broken or unrelated files are not included

## Recommended Manual Run

From the project root, install dependencies first:

```bash
pip install -r requirements.txt
```

If you want ingestion to download the dataset with KaggleHub:

```bash
python scripts/run_ingestion.py --download
```

If you already have the dataset on disk and want to override the config:

```bash
python scripts/run_ingestion.py \
  --dataset-root data/raw/alzheimers_multiclass \
  --output-manifest data/processed/manifest.csv
```

## How To Inspect The Output

After ingestion finishes, check that the manifest exists:

```bash
ls data/processed/manifest.csv
```

Preview the first rows:

```bash
head -n 5 data/processed/manifest.csv
```

Count how many rows were written:

```bash
wc -l data/processed/manifest.csv
```

The file should contain these columns:

- `sample_id`
- `image_path`
- `label_name`
- `label_id`

## Acceptance Criteria

Treat ingestion as successful only if all of these are true:

1. The command exits without an exception.
2. The manifest file is created at the configured output path.
3. The manifest has the expected header: `sample_id,image_path,label_name,label_id`.
4. Every manifest row points to an existing image file.
5. Every `label_name` is one of: `NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`.
6. Every `label_id` matches the configured mapping.
7. There are no obviously broken rows such as empty paths or missing labels.
8. The row count is plausible for the source dataset.

## Quick Sanity Checks

These are useful lightweight checks after a real run:

```bash
python - <<'PY'
import csv
from collections import Counter

with open('data/processed/manifest.csv', newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))

print('rows:', len(rows))
print('labels:', Counter(row['label_name'] for row in rows))
print('missing paths:', sum(1 for row in rows if not row['image_path']))
PY
```

## Interpreting The Result

If automated tests pass but manual validation fails, trust the manual run first. That usually means one of these is wrong:

- the real dataset layout differs from the test fixtures
- the config paths are wrong
- the runtime environment is missing a dependency
- the dataset contains files that the current implementation does not handle

That is exactly why the manual check is worth doing at least once before building on top of ingestion.