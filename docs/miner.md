# Poker44 Miner Guide

Production-facing miner guide for Poker44 subnet `126`.

For released supervised benchmark data, see:

- [Training Benchmark](./training-benchmark.md)

## What Miners Are Solving Today

Poker44 validators currently evaluate miners with behavioral payloads derived from
live Poker44 benchmark tables.

Current production path:

1. live benchmark tables run on Poker44 platform infrastructure;
2. those hands are persisted in platform SQL;
3. `poker44-platform-backend` builds evaluation batches from those hands;
4. the validator fetches the active batch set through `/internal/eval/current`;
5. the validator sends those batches to miners through `DetectionSynapse`;
6. miners return one risk score per received chunk;
7. the validator scores the miner and sets weights on-chain.

Important: the miner does **not** receive labels.

Current competition framing:

- competition epochs run for `120h`;
- active canonical evaluation windows run in daily `24h` blocks inside that epoch;
- miners are still evaluated continuously through the same chunk-level contract.

Released benchmark data is different:

- the public benchmark is for historical replay and miner iteration;
- live validator runtime keeps labels on the validator side only.

## Current Miner Contract

Miners receive `DetectionSynapse(chunks=...)`.

Current semantics:

- `chunks` is a list of chunks;
- each chunk is a list of hand payloads;
- each chunk may contain one or many hands;
- the validator expects exactly one `risk_score` per chunk.

So today the practical task is:

- receive many chunks per request;
- score each chunk independently;
- return one probability-like bot score per chunk.

Relevant code:

- [DetectionSynapse](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/synapse.py)
- [reference miner](/Users/mac/poker44-launch/poker44-subnet/neurons/miner.py)
- [validator forward cycle](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/forward.py)

## Important Precision About Chunk Structure

There are two different layers:

1. source hands on benchmark tables
2. chunks delivered to miners

Today, platform source hands are collected from live benchmark tables where humans and bots sit
together.

But the chunk format delivered to miners is still aligned with the current scoring path:

- the backend builds chunk-level evaluation batches from benchmark-table hands;
- the validator groups those batches into `DetectionSynapse(chunks=...)`;
- miners return one score per batch/chunk.

So:

- the overall validator request can contain multiple chunk types;
- each individual chunk remains one scoring unit from the validator’s point of view;
- miners should not assume a fixed number of hands per chunk.

Do not build your miner assuming this exact granularity will never evolve, but document and
optimize against the contract that is live today: one score per received chunk.

## Payload Shape

The payload sent to miners is a backend-prepared evaluation payload.

The miner-visible hand structure includes:

- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`

Important: the miner does not receive explicit ground-truth labels.

## Expected Miner Output

Return fields:

- `risk_scores: List[float]`
- `predictions: List[bool]` optional but recommended
- `model_manifest: Dict[str, Any]` optional but recommended

Rules:

- length of `risk_scores` must equal number of received chunks;
- each score should be in `[0, 1]`;
- `predictions` should align one-to-one with `risk_scores` when provided.

The reference miner treats each chunk as one scoring unit and returns:

- low score for human-like behavior
- high score for bot-like behavior

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use:

```bash
./scripts/miner/setup.sh
```

## Wallet and Registration

`btcli` is provided by the separate `bittensor-cli` package.

```bash
btcli wallet new_coldkey --wallet.name my_cold
btcli wallet new_hotkey --wallet.name my_cold --wallet.hotkey my_poker44_hotkey

btcli subnet register \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name my_cold --subtensor.network finney
```

## Run Miner

Script path:

- `scripts/miner/run/run_miner.sh`

Example:

```bash
WALLET_NAME=my_cold \
HOTKEY=my_poker44_hotkey \
AXON_PORT=8091 \
ALLOWED_VALIDATOR_HOTKEYS="validator_hotkey_1 validator_hotkey_2" \
./scripts/miner/run/run_miner.sh
```

Before using the script, set at least:

- `WALLET_NAME`
- `HOTKEY`
- `AXON_PORT`
- `ALLOWED_VALIDATOR_HOTKEYS` for the recommended allowlist mode

If `ALLOWED_VALIDATOR_HOTKEYS` is empty, the script falls back to
`--blacklist.force_validator_permit`.

Direct CLI example:

```bash
python neurons/miner.py \
  --netuid 126 \
  --wallet.name my_cold \
  --wallet.hotkey my_poker44_hotkey \
  --subtensor.network finney \
  --axon.port 8091 \
  --blacklist.allowed_validator_hotkeys <validator_hotkey_1> <validator_hotkey_2>
```

## Production Access Policy

Recommended mode:

- `--blacklist.allowed_validator_hotkeys <validator_hotkey...>`

Fallback mode:

- `--blacklist.force_validator_permit`

Operationally:

- if an allowlist is set, only those validators may query your miner;
- otherwise the miner falls back to the metagraph `validator_permit` rule.

## Model Manifest

Poker44 miners can publish a lightweight `model_manifest` without changing the remote-inference
scoring path.

Recommended fields:

- `schema_version`
- `open_source`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `framework`
- `license`
- `training_data_statement`
- `training_data_sources`
- `private_data_attestation`
- `data_attestation`
- `artifact_url`
- `artifact_sha256`
- `model_card_url`
- `inference_mode`
- `implementation_files`
- `implementation_sha256`
- `notes`

Current production minimum for `transparent` compliance:

- `open_source=true`
- `repo_url`
- `repo_commit`
- `model_name`
- `model_version`
- `training_data_statement`
- `private_data_attestation`
- `implementation_files`
- `implementation_sha256`

Current production policy notes:

- `repo_commit` should be a real git commit hash, not a placeholder;
- `repo_url` should point to the actual public model repo, not a generic reference repo;
- `implementation_files` should identify the implementation files backing the miner;
- `implementation_sha256` should match the code actually being served;
- you should also publish `data_attestation` where applicable, even if a given compliance surface
  is primarily checking `private_data_attestation`.

The validator still scores your `risk_scores`; the manifest is for transparency and
runtime tracking.

Important:

- once a model reaches a sufficiently high score, its published manifest may be reviewed more deeply;
- the published repository and commit must make the full model flow publicly visible and verifiable;
- if the published repo/commit does not match the logic behind the observed performance, the model
  may be penalized, disqualified, or reduced to a score of `0`;
- incomplete or inconsistent public model identity can trigger stricter review and compliance action.

## Production Evaluation Boundary

Production evaluation is not derived from local helper artifacts.

Production validators now target:

- live hands generated on Poker44 platform tables
- centralized SQL persistence
- batches served by the eval API

Miners should optimize against the live contract and the current chunk-level scoring path, not
against assumptions about any local reference corpus.

## Public Training Benchmark

Poker44 also exposes a public benchmark through the backend API.

That benchmark:

- is intended as a public training surface for miner iteration;
- exposes the miner-visible chunk payload;
- exposes label data separately from the hand payload;
- is deliberately separated from the live competition surface.

Current production status:

- the benchmark API is live on `https://api.poker44.net/api/v1/benchmark`;
- release discovery is available through `GET /api/v1/benchmark/releases`;
- chunk discovery is available through `GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`;
- miners should check the benchmark status endpoint instead of assuming a fixed release cadence,
  because benchmark publication policy may be adjusted operationally.

See:

- [Training Benchmark](./training-benchmark.md)

## PM2

```bash
pm2 logs poker44_miner
pm2 restart poker44_miner
pm2 stop poker44_miner
pm2 delete poker44_miner
```

## Health Checklist

- Miner hotkey registered on netuid `126`
- Axon served and reachable
- Validator queries accepted
- Returned `risk_scores` length matches chunk count
- Miner remains stable under repeated validator polling
