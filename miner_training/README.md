# Poker44 Miner Training Toolkit

Pull & parse the public training benchmark, train a bot-detector, and serve it
from the miner with **zero train/serve skew**. Everything here is scored against
the subnet's *own* reward function, not generic accuracy.

## What you are optimizing (read this first)

The on-chain reward (see [`poker44/score/scoring.py`](../poker44/score/scoring.py)) is:

```
reward = 0.75 * average_precision + 0.25 * (bot recall @ FPR <= 5%)
```

computed over a rolling window of chunk scores, and **emissions are winner-take-all**
(only the single highest-reward miner earns — see
[`poker44/validator/forward.py`](../poker44/validator/forward.py) `_select_weight_targets`).

Consequences that shape the whole approach:

1. **It's a ranking problem, not calibration.** Only the *order* of your chunk
   scores matters. The validator sweeps the threshold itself; you just need bot
   chunks to rank above human chunks, with a clean high-score region (that's what
   drives recall @ FPR≤5%).
2. **Generalization is the entire game.** You must beat *every* other miner on
   *unseen, evolving* bots. A model that memorizes one release date loses. Select
   models on **held-out-by-date** reward, never on train reward.
3. **Second place earns nothing.** Small, robust edges compound into the win.

## Key facts verified against the live API

- **Train/serve parity holds.** Benchmark hands are already in the exact
  sanitized, miner-visible form the validator serves (`bb=0.02`,
  `hole_cards=null`, seats aliased to `seat_N`, `outcome` zeroed, actions
  down-sampled). So you can train directly on benchmark hands. `features.py`
  assumes this form and is imported by both trainer and miner.
- **Each chunk is labeled for one "hero."** `metadata.hero_seat` identifies the
  hero's aliased seat in every hand. **Isolating the hero's own actions is the
  main edge** — the reference miner blindly averages all players. `features.py`
  filters `actor_seat == hero_seat`.
- **No timing signal.** Decision-time fields are stripped from the miner payload.
  Do not build timing features — they cannot exist at serve time.

## Files

| file | purpose |
|------|---------|
| `benchmark_client.py` | fetch status/releases/chunks (+`download_all`), cache by `chunkHash`, parse to `(hands, label, meta)` |
| `features.py`         | chunk → 85-dim feature vector (hero-centric behavior + sizing discreteness); **imported by trainer AND miner** |
| `ensemble.py`         | picklable recency-weighted ensemble (LightGBM + RF + ExtraTrees + HistGBM) |
| `train.py`            | train + evaluate with the exact subnet `reward()`; hold out by date; recency-weight older dates |
| `cv.py`               | **honest local score**: cross-validated subnet reward (same-version CV / holdout / LODO) |
| `serve.py`            | load model, `score_chunks()` for the miner (falls back to 0.5 if no model) |
| `test_local.py`       | verify the serve path + output contract on a chosen date |

## Quick start

```bash
# from repo root, with the project venv
python -m miner_training.benchmark_client        # smoke: show latest date + releases
python -m miner_training.train --all             # download ALL dates, train, refit deploy model -> model.pkl
python -m miner_training.cv                       # honest cross-validated local score (the real number)
python -m miner_training.test_local               # verify serve path + output contract
```

`--all` downloads every release date (cached under `benchmark_cache/`, git-ignored),
holds out the newest date for a quick estimate, then refits the deploy model on
everything. `--half-life 10` controls recency weighting (older bot generations
count exponentially less). LightGBM is used if installed (`pip install lightgbm`);
otherwise it falls back to an sklearn-only ensemble.

Then wire it into [`neurons/miner.py`](../neurons/miner.py):

```python
from miner_training.serve import ModelScorer
# in __init__:
self.scorer = ModelScorer("miner_training/model.pkl")
# in forward():
scores = self.scorer.score_chunks(synapse.chunks or [])
synapse.risk_scores = scores
synapse.predictions = [s >= 0.5 for s in scores]
synapse.model_manifest = dict(self.model_manifest)
```

## Reference result (verified end-to-end, all 42 release dates, 724 groups)

Cross-validated with the exact subnet reward (`python -m miner_training.cv`,
10-day recency half-life). No group is ever scored by a model that trained on it.

| view | what it measures | reward | AP | recall@FPR5% |
|------|------------------|--------|----|--------------|
| **same-version CV** | deploy-like: recent training data, score fresh same-version hands | **0.709** | 0.809 | 0.409 |
| cross-version holdout | pessimistic: newest date is the only one of its bot version | 0.689 | 0.796 | 0.366 |
| leave-one-date-out | conservative lower bound across all dates | 0.666 | 0.774 | 0.340 |
| reference heuristic (shipped) | the baseline you are replacing | 0.458 | 0.564 | 0.141 |

What moved the number from the first cut (0.65) to 0.71:
1. **recency weighting** (older v1.12 dates down-weighted vs current v1.13): +~0.02, the step over 0.70;
2. **v3 hero features** (sizing discreteness, pot-pressure reactions, per-street sizing): +~0.02;
3. **regularized bagged ensemble** instead of one over-fit GBM: +~0.03 and far less variance;
4. **more data** (1 date → 42 dates): +~0.14 on the held-out estimate.

`recall@FPR5%` is the bottleneck term and is small-sample-noisy here (only 71
humans in the test window). On live-scale batches the 5%-FPR threshold is far
smoother, so the live reward for an AP≈0.81 model is typically **at or above** this
estimate. The single biggest further lever is simply **more recent same-version
data** — retrain as new releases land.

## Roadmap to the top (in priority order)

1. **Accumulate data.** 142 groups/date overfits a 57-feature model. New releases
   ship ~daily — pull the full `/releases` history and every new date, cache it,
   and retrain. More labeled groups is the single biggest lever right now.
2. **Select on cross-date reward.** Use date-grouped CV (leave-one-date-out). If a
   feature/model helps train but not held-out dates, drop it.
3. **Regularize hard** while data is scarce (fewer leaves, higher `min_samples_leaf`
   / `reg_lambda`, early stopping on a held-out date).
4. **Deepen hero features.** Bet-sizing *discreteness* (low CV / low entropy =
   robotic), bet-to-pot regularity per street, action-sequence n-grams, reaction
   to facing a bet/raise, cross-hand uniformity. Keep everything hero-centric and
   parity-safe.
5. **Stronger models once data supports it.** LightGBM/XGBoost (auto-used if
   installed), then permutation-invariant set models over per-hand embeddings
   (DeepSets / attention pooling) — a chunk is a *set* of hands. Ensemble + rank
   averaging for a stable top-of-list.
6. **Optimize the actual metric.** Consider a ranking objective (LambdaMART /
   `rank:pairwise`) or AP-surrogate; and since 25% of reward is recall@FPR≤5%,
   push purity at the very top of your score distribution.
7. **Retrain on schedule.** Bots evolve; a stale model decays. Automate daily pull
   + retrain + redeploy, and watch per-release reward for drift.

## Guardrails

- **Never skew train vs serve.** Both must import this exact `features.py`. If you
  change features, retrain before deploying (`serve.py` refuses a mismatched model).
- **No leakage features:** never use `chunkId`, `chunkHash`, `hand_id`, dates,
  release version, seat numbers, or group order as inputs.
- **Don't overfit one date.** Train and validate across multiple release dates.
- **Publish an honest manifest** (`repo_url`, real `repo_commit`,
  `implementation_sha256`). High scorers may be audited; a mismatch between your
  public repo and served logic can be penalized or zeroed (see `docs/miner.md`).
- **Always return `len(risk_scores) == len(chunks)`**, each in `[0,1]`, or the
  validator discards the whole response.
```
