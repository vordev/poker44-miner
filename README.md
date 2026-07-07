<div align="center">
  <h1>🂡 <strong>Poker44</strong> — Poker Bot Detection Subnet</h1>
  <img src="poker44/assets/logopoker44.png" alt="Poker44 logo" style="width:320px;">
  <p>
    <a href="docs/validator.md">🔐 Validator Guide</a> &bull;
    <a href="docs/miner.md">🛠️ Miner Guide</a> &bull;
    <a href="docs/training-benchmark.md">📦 Training Benchmark</a> &bull;
    <a href="https://poker44.net">🌐 Platform</a>
  </p>
</div>

---

## Official Links

- X: https://x.com/poker44subnet
- Web: https://poker44.net
- Whitepaper: https://poker44.net/Poker44_Whitepaper.pdf

---

## What is Poker44?

Poker44 is a Bittensor subnet focused on one problem: detecting bots in online poker with
objective, reproducible evaluation.

Validators query miners with poker-behavior chunk payloads, score predictions, and publish
weights on-chain. Miners compete by returning robust bot-risk predictions that generalize to
evolving live-table behavior.

Poker44 is security infrastructure, not a poker room.

---

## Current Production Model

The current production direction is:

- evaluation data is produced by Poker44 platform infrastructure;
- validators do **not** run their own tables;
- validators consume canonical evaluation material from the central eval API;
- validators query miners, score responses, and set weights on-chain.

The production path also carries the observability layer used by the public
competition surfaces on `poker44-platform-*`.

The validator production path is now the central `provider_runtime` model.

---

## What Miners Receive Today

Miners receive `DetectionSynapse(chunks=...)`.

Current semantics:

- `chunks` is a list of chunks;
- each chunk is a list of hand payloads;
- validators expect one `risk_score` per chunk;
- each chunk may contain one or many hands.

This means:

- miners should treat each chunk as one scoring unit, regardless of how many hands it contains.

The competition framing should be understood as:

- time-based competition epochs;
- continuous evaluation on canonical live batches;
- public leaderboard surfaces derived from signed runtime state.

At the current production cadence:

- competitions run in rolling `120h` epochs anchored at the active cycle close;
- canonical evaluation material is published and refreshed in daily `24h` windows;
- the latest fully settled competition winner remains the canonical reference slot until the next settlement closes.

In the current runtime, validators read the canonical competition weight
vector from the backend. Competition policy and allocation rules are determined
by the platform runtime and may evolve independently of the reference code in
this repo.

See:

- [Miner Guide](docs/miner.md)
- [Validator Guide](docs/validator.md)
- [Training Benchmark](docs/training-benchmark.md)

---

## Data Model Boundary

Production validators now target:

- centralized evaluation data supplied by Poker44 platform infrastructure.

The repo may still include reference tooling for miner development, but production evaluation is
driven by the central platform runtime and should not be inferred from local helper artifacts.

---

## Open-Source Miner Standard

Poker44 supports a lightweight `model_manifest` attached to miner responses.

This does not change validator scoring or on-chain `set_weights`. It adds:

- traceability
- training-data disclosure
- transparency metadata
- evaluation observability

Recommended manifest fields include:

- repo URL
- repo commit or tag
- model name and version
- framework
- license
- training-data statement
- data-handling attestation

---

## Quick Start

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then follow:

- [Validator setup](docs/validator.md)
- [Miner setup](docs/miner.md)
- [Training benchmark](docs/training-benchmark.md)

---

## Repository Links

- Validator docs: [`docs/validator.md`](docs/validator.md)
- Miner docs: [`docs/miner.md`](docs/miner.md)
- Training benchmark docs: [`docs/training-benchmark.md`](docs/training-benchmark.md)

---

## License

MIT — see [`LICENSE`](LICENSE).
