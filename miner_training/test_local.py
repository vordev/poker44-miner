"""Local test harness for the trained miner model.

Exercises the EXACT serve path the miner uses (`ModelScorer.score_chunks`) over
real cached benchmark chunks, verifies the output contract the validator enforces
(one score per chunk, each in [0, 1]), and reports the subnet reward vs the
ground-truth labels -- alongside the reference heuristic baseline.

    python -m miner_training.test_local                    # newest cached date
    python -m miner_training.test_local --date 2026-07-06
    python -m miner_training.test_local --model miner_training/model.pkl --show 15
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np

from poker44.score.scoring import reward
from miner_training import benchmark_client as bc
from miner_training.serve import ModelScorer


# --- reference heuristic (mirrors neurons/miner.py, no bittensor dependency) ---
def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _ref_score_hand(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}
    ac = Counter(a.get("action_type") for a in actions)
    m = max(1, sum(ac.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
    call_r, check_r, fold_r, raise_r = ac.get("call", 0) / m, ac.get("check", 0) / m, ac.get("fold", 0) / m, ac.get("raise", 0) / m
    depth = len(streets) / 3.0
    show = 1.0 if outcome.get("showdown") else 0.0
    pcs = (6 - min(len(players), 6)) / 4.0 if players else 0.0
    s = (0.32 * depth + 0.22 * show + 0.18 * _clamp01(call_r / 0.35) + 0.12 * _clamp01(check_r / 0.30)
         + 0.08 * _clamp01(pcs) - 0.18 * _clamp01(fold_r / 0.55) - 0.10 * _clamp01(raise_r / 0.20))
    return _clamp01(s)


def _ref_chunk(chunk: List[dict]) -> float:
    if not chunk:
        return 0.5
    return round(_clamp01(sum(_ref_score_hand(h) for h in chunk) / len(chunk)), 6)


def _cached_dates(cache_dir: str) -> List[str]:
    root = Path(cache_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load_groups(date: str, cache_dir: str) -> Tuple[List[List[dict]], List[int]]:
    records = bc.load_cached_date(date, cache_dir) or bc.get_date(date, cache_dir=cache_dir)
    groups: List[List[dict]] = []
    labels: List[int] = []
    for hands, label, _meta in bc.iter_examples(records):
        groups.append(hands)
        labels.append(int(label))
    return groups, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="miner_training/model.pkl")
    ap.add_argument("--date", default="", help="release date to test on (default: newest cached)")
    ap.add_argument("--cache-dir", default=bc.DEFAULT_CACHE)
    ap.add_argument("--show", type=int, default=12, help="sample predictions to print")
    args = ap.parse_args()

    date = args.date.strip()
    if not date:
        cached = _cached_dates(args.cache_dir)
        if not cached:
            raise SystemExit(f"No cached data in {args.cache_dir!r}; run `python -m miner_training.train --all` first.")
        date = cached[-1]

    groups, labels = _load_groups(date, args.cache_dir)
    if not groups:
        raise SystemExit(f"No labeled groups for {date}.")
    y = np.asarray(labels, dtype=int)

    scorer = ModelScorer(args.model)
    if scorer.model is None:
        print(f"WARNING: no model at {args.model!r}; scoring with 0.5 fallback.\n")

    # ---- exact serve path ----
    scores = scorer.score_chunks(groups)

    # ---- contract checks (what the validator enforces) ----
    problems = []
    if len(scores) != len(groups):
        problems.append(f"length mismatch: {len(scores)} scores for {len(groups)} chunks")
    if not all(isinstance(s, float) for s in scores):
        problems.append("non-float score present")
    if not all(0.0 <= s <= 1.0 for s in scores):
        problems.append("score outside [0, 1]")
    contract = "PASS" if not problems else "FAIL -> " + "; ".join(problems)

    print(f"Test date: {date}  |  chunks: {len(groups)}  |  bot/human: {int(y.sum())}/{int((y == 0).sum())}")
    print(f"Output contract (one score per chunk, each in [0,1]): {contract}\n")

    model_rew, m = reward(np.asarray(scores, dtype=float), y)
    ref_scores = [_ref_chunk(g) for g in groups]
    ref_rew, r = reward(np.asarray(ref_scores, dtype=float), y)

    print("Local performance (subnet reward = 0.75*AP + 0.25*recall@FPR5%):")
    print(f"  trained model     : reward={model_rew:.4f}  AP={m['ap_score']:.4f}  recall@fpr5%={m['bot_recall']:.4f}")
    print(f"  reference heuristic: reward={ref_rew:.4f}  AP={r['ap_score']:.4f}  recall@fpr5%={r['bot_recall']:.4f}")
    print(f"  uplift            : {model_rew - ref_rew:+.4f} reward\n")
    print("NOTE: model.pkl (from `train --all`) trains on every cached date, so this score is\n"
          "      IN-SAMPLE (optimistic). For the honest cross-validated reward, run:\n"
          "          python -m miner_training.cv\n")

    order = np.argsort(-np.asarray(scores))
    print(f"Sample predictions (top {min(args.show, len(scores))} by model risk):")
    print(f"  {'truth':6} {'model':>7} {'ref':>7}")
    for i in order[: args.show]:
        print(f"  {'BOT' if y[i] == 1 else 'human':6} {scores[i]:7.4f} {ref_scores[i]:7.4f}")


if __name__ == "__main__":
    main()
