# Poker44 Training Benchmark

Public benchmark guide for Poker44 subnet `126`.

## Purpose

Poker44 provides a public training benchmark for miner development. Use it to:

- test your benchmark parser;
- build and validate feature pipelines;
- train and compare detection models;
- run regression tests across model versions;
- calibrate model outputs against labeled chunk data.

The benchmark is a development dataset for model training, validation, parser
testing, and regression testing.

## API Base

```text
https://api.poker44.net/api/v1/benchmark
```

The benchmark API is public and does not require authentication.

## Endpoints

```text
GET /api/v1/benchmark
GET /api/v1/benchmark/releases
GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD
GET /api/v1/benchmark/chunks/:chunkId
```

There is no separate `/training/latest` endpoint. To use the latest benchmark,
first call the status endpoint, read `latestSourceDate`, then request chunks for
that date.

## Latest Benchmark

`GET /api/v1/benchmark` returns aggregate availability and the latest published
training benchmark date.

- `releaseVersion`
- `schemaVersion`
- `releaseType`
- `totalChunks`
- `totalHands`
- `latestSourceDate`
- `latestReleasedAt`
- `currentUtcDate`
- `minimumHumanExamplesPerRelease`
- `targetHumanExamplesPerRelease`
- `defaultChunkLimit`
- `autoRelease`

Example:

```bash
curl -sS https://api.poker44.net/api/v1/benchmark
```

Use `latestSourceDate` from this response as the `sourceDate` for chunk
downloads.

## Releases

`GET /api/v1/benchmark/releases` returns the history of published training
benchmark releases.

Common query parameters:

- `limit`: number of releases to return.
- `before`: optional `YYYY-MM-DD` cursor for pagination.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/releases?limit=30'
```

Each release includes:

- `sourceDate`
- `releaseVersion`
- `schemaVersion`
- `chunkCount`
- `handCount`
- `releasedAt`
- `humanExampleCount`
- `syntheticBotExampleCount`
- `audit`
- `metadata`

Use releases to compare model behavior across benchmark versions. For normal
development, start from the latest `sourceDate`.

## Chunks

`GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD` returns chunk payloads for
one release date.

Common query parameters:

- `sourceDate`: required release date in `YYYY-MM-DD` format.
- `limit`: number of chunks to return.
- `cursor`: optional pagination cursor.
- `split`: optional `train` or `validation`.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-07-06&limit=24'
```

Each chunk includes:

- `chunkId`
- `chunkHash`
- `chunkIndex`
- `sourceDate`
- `releaseVersion`
- `split`
- `handCount`
- `batchCount`
- `chunkCount`
- `chunks`
- `groundTruth`
- `groundTruthLabels`
- `metadata`

## Model Input

The `chunks` field is the miner-visible model input. It is a list of chunk
groups. Each group contains one or more poker hands.

Miners should produce one prediction per chunk group, matching the order of
`chunks`.

Current releases use at least 30 hands per chunk group.

The labels are returned separately:

- `groundTruth`: numeric labels, where `1` means bot and `0` means human.
- `groundTruthLabels`: string labels, `bot` or `human`.

Do not read labels from individual hand objects.

Minimal validation example:

```python
import requests

base_url = "https://api.poker44.net/api/v1/benchmark"

status = requests.get(base_url, timeout=30).json()["data"]
source_date = status["latestSourceDate"]

payload = requests.get(
    f"{base_url}/chunks",
    params={"sourceDate": source_date, "limit": 100},
    timeout=30,
).json()["data"]

for chunk in payload["chunks"]:
    model_inputs = chunk["chunks"]
    labels = chunk["groundTruth"]

    predictions = model.predict_proba(model_inputs)

    assert len(predictions) == len(labels)
    assert all(0.0 <= score <= 1.0 for score in predictions)
```

Paginated download example:

```python
import requests

base_url = "https://api.poker44.net/api/v1/benchmark"
source_date = requests.get(base_url, timeout=30).json()["data"]["latestSourceDate"]

all_chunks = []
cursor = None

while True:
    params = {"sourceDate": source_date, "limit": 24}
    if cursor:
        params["cursor"] = cursor

    data = requests.get(f"{base_url}/chunks", params=params, timeout=60).json()["data"]
    all_chunks.extend(data["chunks"])
    cursor = data.get("nextCursor")
    if not cursor:
        break

print(f"downloaded {len(all_chunks)} chunks for {source_date}")
```

## Hand Fields

Hands may include:

- `hand_id`
- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`

Action records may include:

- `action_id`
- `street`
- `actor_seat`
- `action_type`
- `amount`
- `raise_to`
- `call_to`
- `normalized_amount_bb`
- `pot_before`
- `pot_after`

Code should tolerate missing optional fields and empty arrays.

## Training Guidance

Use each chunk group as one training example. The target is the matching entry
in `groundTruth`, in the same array position.

Recommended practices:

- keep release dates separate when creating train, validation, and local test
  sets;
- use the returned `split` field when it is present;
- train across multiple release dates instead of fitting one date tightly;
- save `sourceDate`, `releaseVersion`, `schemaVersion`, `chunkId`, and
  `chunkHash` with every local experiment;
- cache downloaded JSON by `chunkHash` so experiments are reproducible;
- ignore unknown response fields so clients keep working as the schema expands;
- avoid using identifiers such as `hand_id` or `chunkId` as model features.

Useful metrics:

- ROC AUC for ranking quality;
- average precision for bot-class retrieval;
- log loss or Brier score for probability calibration;
- per-release metrics to catch overfitting to one benchmark date.

## Recommended Workflow

1. Fetch status from `/api/v1/benchmark`.
2. Read `latestSourceDate`.
3. Download chunks with `/api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`.
4. Cache raw responses and record `chunkHash`.
5. Split by release date and by the returned `split` field when present.
6. Build features only from miner-visible hand and action data.
7. Train on `train` chunks.
8. Tune and compare on `validation` chunks.
9. Keep a held-out local set for model regression tests.
10. Track performance by release date and model version.

## Common Mistakes

- Producing one prediction per hand instead of one prediction per chunk group.
- Reordering `chunks` before pairing predictions with `groundTruth`.
- Training and validating on the same release date only.
- Treating optional fields as always present.
- Using IDs, dates, hashes, or pagination order as predictive features.
- Assuming every chunk group has the same number of hands or actions.

## Notes

- New releases may be added over time.
- The status endpoint is the source of truth for the latest public benchmark
  date.
- The releases endpoint is for history and comparison across benchmark
  versions.
- Response fields may expand, so clients should ignore unknown fields.
- The chunk order and label order are significant.
- Avoid tuning a model against a single release only.
- Prefer testing across multiple release dates.
- Use the benchmark for model development, parser testing, feature validation,
  and regression testing.
