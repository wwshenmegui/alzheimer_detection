# Feature Layer Manual Validation

Automated tests are necessary for the feature layer, but you should still run one manual end-to-end check on the real validated manifest before using the outputs for model training.

## What The Feature Layer Does

The current feature layer is intentionally simple.

It does only this:

- reads `data/processed/validated_manifest.csv`
- loads each image from disk
- converts it to grayscale
- resizes it to a fixed size
- normalizes pixel values to `[0, 1]`
- assigns an explicit `train` or `validation` or `test` split using a `6:2:2` ratio
- saves all images, labels, sample IDs, and split labels into a compressed NumPy artifact

## How To Run It Manually

From the project root:

```bash
pip install -r requirements.txt
python scripts/run_features.py
```

That uses the paths and image size defined in `configs/training.yaml`.

If you want to override them:

```bash
python scripts/run_features.py \
  --validated-manifest data/processed/validated_manifest.csv \
  --output-features data/processed/features.npz \
  --output-report data/reports/features_report.json \
  --image-width 128 \
  --image-height 128
```

## Expected Output Files

After a successful run, you should see:

- `data/processed/features.npz`
- `data/reports/features_report.json`

## How To Inspect The Result

Check that the files exist:

```bash
ls data/processed/features.npz
ls data/reports/features_report.json
```

Print the report:

```bash
cat data/reports/features_report.json
```

Inspect the NumPy artifact:

```bash
python - <<'PY'
import numpy as np

artifact = np.load('data/processed/features.npz')
print('images shape:', artifact['images'].shape)
print('labels shape:', artifact['labels'].shape)
print('sample_ids shape:', artifact['sample_ids'].shape)
print('splits shape:', artifact['splits'].shape)
print('pixel min/max:', artifact['images'].min(), artifact['images'].max())
PY
```

## Acceptance Criteria

Treat the feature layer as successful only if all of these are true:

1. The command exits without an exception.
2. `data/processed/features.npz` is created.
3. `data/reports/features_report.json` is created.
4. The report contains `passed: true`.
5. The report contains a positive `total_rows` value.
6. The report contains `image_shape` equal to the configured grayscale tensor shape.
7. The `images` array has shape `(N, H, W, 1)`.
8. The `labels` array length matches the number of images.
9. The `splits` array exists and contains `train`, `validation`, and `test`.
10. The split ratio is approximately `6:2:2`.
11. Pixel values are normalized to the range `[0, 1]`.

## Failure Criteria

Treat the feature layer as failed if any of these happen:

- the validated manifest file is missing
- required manifest columns are missing
- the validated manifest is empty
- an image file cannot be opened during preprocessing
- the output feature artifact is missing after a reported success
- image tensors are not grayscale 1-channel outputs

If the report and the saved artifact both match the expected structure, the current feature layer is working as intended.
