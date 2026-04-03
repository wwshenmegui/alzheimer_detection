# Serving Module Manual Validation

Automated tests are necessary for the serving module, but you should still run one manual end-to-end check with the real trained model before treating the API as ready.

## What The Serving Module Does

The current serving module is intentionally simple.

It does only this:

- loads `models/trained/logistic_regression.pkl`
- accepts one uploaded MRI image
- applies the same grayscale preprocessing used in training
- runs model inference
- returns the predicted label and per-class probabilities

## How To Run It Manually

From the project root:

```bash
pip install -r requirements.txt
python scripts/run_server.py
```

That uses the serving settings in `configs/training.yaml`.

If you want to override host, port, or log level:

```bash
python scripts/run_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --log-level INFO
```

## Manual Checks

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/sample.png"
```

## Acceptance Criteria

Treat serving as successful only if all of these are true:

1. The server starts without an exception.
2. `/health` returns HTTP `200` and `status: ok`.
3. `/predict` accepts an image upload and returns HTTP `200`.
4. The prediction response contains `predicted_label`, `predicted_label_id`, `probabilities`, `input_shape`, and `model_name`.
5. The response probabilities cover all four labels.
6. The reported input shape matches training preprocessing.

## Failure Criteria

Treat serving as failed if any of these happen:

- the model file cannot be loaded
- the server does not start
- `/health` fails
- `/predict` rejects valid images
- `/predict` returns malformed JSON
- preprocessing shape does not match training expectations

If `/health` and `/predict` both behave as expected, the current serving module is working as intended.
