# Poker44 Validator Guide

Validator guide for Poker44 subnet `126`.

## Current Architecture

Poker44 validators are now intended to run in a **consumer-only** model.

That means:

- validators do **not** run their own poker tables;
- validators do **not** bootstrap provider frontend/backend locally;
- validators do **not** build live evaluation data from local JSON on the production path;
- validators consume canonical evaluation batches from the central Poker44 eval API;
- validators query miners, compute rewards, and set weights on-chain.

The validator production path is now `provider_runtime` only.

## Separation of Responsibilities

### Poker44 platform infrastructure owns

- the live benchmark tables;
- bots seated at those tables;
- real-time gameplay;
- SQL persistence of hands and events;
- chunk publication through `/internal/eval/*`.

### `poker44-subnet` validator owns

- polling the eval API;
- fetching the active canonical batch set;
- querying miners;
- scoring miner responses;
- updating weights on-chain;
- marking evaluated batch refs back to the API.

This is the key design boundary: live table/runtime logic lives in `poker44-platform-*`, not in
the validator.

## What the Validator Actually Sends to Miners

The validator fetches `batches` from the central eval API. Each returned batch currently looks like:

- one hidden label (`is_human`) on the validator side only;
- one list of `hands`;
- one chunk-sized evaluation unit that may contain one or many hands.

Then the validator converts those batches into:

- `DetectionSynapse(chunks=...)`

Where:

- `chunks` is a list of chunks;
- each chunk is a list of hands;
- each chunk may contain one or many hands;
- miners return one score per chunk.

So the current production path is **not** “one label for the entire epoch payload”.
Instead, the validator scores miners chunk-by-chunk, with each chunk treated as
one scoring unit.

Relevant code:

- [validator entrypoint](/Users/mac/poker44-launch/poker44-subnet/neurons/validator.py)
- [runtime provider](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/runtime_provider.py)
- [forward cycle](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/forward.py)
- [synapse](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/synapse.py)

## Where the Eval Data Comes From

The current production source is centralized platform infrastructure. In broad
terms, live gameplay data is persisted by the platform runtime, transformed into
canonical evaluation material by the backend, and then consumed by validators
through the eval API.

## Observability And Competition Signals

The validator also publishes two signed observability payloads:

- `validator_runtime.json`
- `network_snapshot.json`

These are best-effort and are not part of the scoring path. They exist so the
platform can expose:

- validator runtime alignment;
- live network/miner state from validator-signed metagraph snapshots;
- a daily competition surface built on top of the canonical eval feed.

The intended competition model is time-based and continuously evaluated, with
public leaderboard surfaces derived from signed validator and network state.

At the current production cadence:

- competition epochs run for `120h`, anchored at the active cycle close;
- canonical eval chunks are managed in daily `24h` windows inside that epoch;
- the latest fully settled competition winner remains the canonical competition reference until the next settlement closes.

Settlement behavior now follows a platform-decided pattern:

- validators fetch the canonical competition vector from
  `/internal/competition/current/weights`;
- once the backend has settled at least one competition winner, the latest settled
  winner becomes the canonical competition vector for the current/vigente
  period, but validators apply a Swarm-style burn on top of it:
  `0%` to `uid 0`, `100%` to the backend-provided winner vector;
- before the first settlement exists, the backend returns its explicit
  fallback vector (typically `uid 0`, which remains `100%` burned);
- validators only fall back to local score-based weights if the backend is
  unavailable or returns no usable positive vector.

Important nuance:

- validators poll the runtime continuously;
- the active canonical chunk can be refreshed over time;
- each delivered chunk remains one scoring unit from the validator’s point of view.

## Pull + Restart Contract

When a validator operator does only:

1. `git pull`
2. restart the validator

the validator should resume normal evaluation against the central eval API.

Concretely:

- it starts in `provider_runtime`;
- it connects to the central Poker44 eval API;
- it checks whether enough real hands exist;
- it may request publication of the current canonical chunk;
- it fetches the active chunk;
- it sends that chunk set to miners;
- it computes rewards;
- it sets weights;
- it marks evaluated batch refs back to the API.

## Requirements

- Linux server
- Python 3.10+
- PM2
- registered validator hotkey on netuid `126`
- network access to the central Poker44 eval API

No local provider stack is required in the target production model.

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
./scripts/validator/main/setup.sh
```

## Registration

```bash
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name p44_cold --subtensor.network finney
```

## Runtime Modes

- `POKER44_RUNTIME_MODE=provider_runtime`

The validator now runs only in `provider_runtime` and consumes central eval data.

## Required Environment

Required for production:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `WALLET_NAME`
- `HOTKEY`

Defaulted for production:

- `POKER44_EVAL_API_BASE_URL=https://api.poker44.net`

Optional observability/reporting:

- `POKER44_VALIDATOR_RUNTIME_REPORT_URL`
- `POKER44_VALIDATOR_NETWORK_SNAPSHOT_REPORT_URL`

Optional audit lane:

- `POKER44_AUDIT_PROVIDER=none|verathos`
- `POKER44_AUDIT_MODE=shadow|disabled`
- `POKER44_AUDIT_TOP_ROWS=8`
- `POKER44_AUDIT_RECENT_REPORT_LIMIT=32`
- `POKER44_VERATHOS_API_KEY`
- `POKER44_VERATHOS_MODEL`
- `POKER44_VERATHOS_BASE_URL=https://api.verathos.ai/v1`
- `POKER44_VERATHOS_TIMEOUT_SECONDS=20`
- `POKER44_AUDIT_PUBLIC_KEY_PEM` (optional override; defaults to the embedded Poker44 public key)

Notes:

- `POKER44_EVAL_API_BASE_URL` points at the central `poker44-platform-backend`;
- `POKER44_PROVIDER_INTERNAL_SECRET` is required for admin eval actions such as
  `/internal/eval/publish-current`;
- validator-facing eval reads and score reporting can run with signed hotkey auth
  when the backend is configured for validator access;
- each batch/chunk may contain one or many hands.

## Audit Lane

Validators now support a best-effort audit lane alongside the main scoring path.

Current behavior:

- the main miner-scoring flow remains unchanged;
- each completed evaluation cycle can produce:
  - `audit_reports.json.enc` with the full audit record encrypted for Poker44;
  - `audit_reports.summary.json` with only non-sensitive local summary fields;
- when `POKER44_AUDIT_PROVIDER=none`, the validator still records an encrypted local audit trail with:
  - epoch/chunk identifiers,
  - dataset hash,
  - top competition rows,
  - validator/runtime context,
  - integrity/compliance summaries;
- when `POKER44_AUDIT_PROVIDER=verathos` and Verathos credentials are configured,
  the validator also performs a shadow external audit call and stores the returned
  verification metadata and structured summary.

This audit lane is intentionally best-effort:

- audit failures do not block miner scoring;
- provider failures are recorded but do not interrupt the validator cycle;
- runtime snapshots now include the latest audit summary so the platform can expose
  audit status separately from reward computation.

The encrypted artifact is written with Poker44's audit public key by default, so a
validator operator can store it locally but cannot decrypt the full report from the
node without the corresponding private key held by Poker44.

## Run Validator

Script path:

- `scripts/validator/run/run_vali.sh`

Example:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_RUNTIME_MODE=provider_runtime \
POKER44_PROVIDER_INTERNAL_SECRET=replace-with-real-shared-secret \
POKER44_EVAL_API_BASE_URL=https://api.poker44.net \
./scripts/validator/run/run_vali.sh
```

If the backend auto-publishes the active chunk, operators may leave
`POKER44_PROVIDER_INTERNAL_SECRET` unset and rely on signed validator hotkey
auth for the validator-facing runtime path.

## Canonical Chunk Lifecycle

The current lifecycle is:

1. live benchmark tables generate real hands;
2. hands are persisted in SQL;
3. backend selects eligible benchmark-table hands;
4. backend builds labeled batches from those hands;
5. backend publishes an active canonical chunk for the current window;
6. validator fetches it through `/internal/eval/current`;
7. validator sends the resulting chunk list to miners;
8. validator scores miner responses against the hidden labels;
9. validator marks the evaluated batch refs back to the eval API.

## Current Scoring Granularity

Current scoring granularity is:

- one returned score per chunk;
- one validator label per chunk;
- one chunk may contain one or many hands.

This matters for miner/operator expectations:

- the live source is benchmark-table gameplay;
- the current validator scoring contract is still chunk-level;
- the chunk-level contract is implemented as `list[list[hand]]`, with one score expected per chunk.

## What the Validator Does Not Do

The production validator does **not**:

- run a local poker table;
- deploy provider frontend/backend;
- manage DNS or TLS;
- manage local SQL/Redis for provider runtime;
- generate production eval data from local JSON.

Those are platform responsibilities.

## PM2

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

## Related Docs

- [Miner guide](./miner.md)
