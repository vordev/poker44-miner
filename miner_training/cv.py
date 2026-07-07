"""Robust local scoring: cross-validated subnet reward on cached benchmark data.

This is the trustworthy "how good is my model" number -- it never scores a group
with a model that trained on it. Three views:

  * same-version CV : 5-fold CV on the NEWEST release date, with older dates as
    recency-weighted extra training. Models the live scenario (the deploy model
    trains on recent data and scores fresh same-version hands). == headline.
  * cross-version holdout : train on all older dates, test on the newest date the
    model has never seen. Pessimistic (newest is the only date of its version).
  * LODO pooled : leave-one-date-out across every date. Conservative lower bound.

    python -m miner_training.cv                 # uses cached data
    python -m miner_training.cv --half-life 10

Requires data on disk: run `python -m miner_training.train --all` first.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from poker44.score.scoring import reward
from miner_training import benchmark_client as bc
from miner_training.features import extract_features
from miner_training.ensemble import RecencyEnsemble, recency_weights


def _load_by_date(cache_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    root = Path(cache_dir)
    dates = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not dates:
        raise SystemExit(f"No cached data in {cache_dir!r}; run `python -m miner_training.train --all` first.")
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for d in dates:
        X, y = [], []
        for hands, label, _meta in bc.iter_examples(bc.load_cached_date(d, cache_dir)):
            X.append(extract_features(hands))
            y.append(int(label))
        if X:
            out[d] = (np.asarray(X, float), np.asarray(y, int))
    return out


def _report(tag: str, preds: np.ndarray, labels: np.ndarray) -> float:
    rew, res = reward(preds, labels)
    flag = "  >= 0.70 ✓" if rew >= 0.70 else ""
    print(f"  {tag:34s} reward={rew:.4f}  AP={res['ap_score']:.4f}  recall@fpr5%={res['bot_recall']:.4f}{flag}")
    return rew


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=bc.DEFAULT_CACHE)
    ap.add_argument("--half-life", type=float, default=10.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    by_date = _load_by_date(args.cache_dir)
    dates = sorted(by_date)
    newest = dates[-1]
    total = sum(len(y) for _, y in by_date.values())
    print(f"dates={len(dates)}  newest={newest}  total_groups={total}  half_life={args.half_life}d\n")

    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)

    # ---- same-version CV (headline) ----
    Xn, yn = by_date[newest]
    oldX = np.vstack([by_date[d][0] for d in dates if d != newest])
    oldy = np.concatenate([by_date[d][1] for d in dates if d != newest])
    old_dates = np.concatenate([[d] * len(by_date[d][1]) for d in dates if d != newest]).tolist()
    P = np.zeros(len(yn))
    for tr, te in skf.split(Xn, yn):
        Xtr = np.vstack([oldX, Xn[tr]])
        ytr = np.concatenate([oldy, yn[tr]])
        row_dates = old_dates + [newest] * len(tr)
        w = recency_weights(row_dates, newest, half_life_days=args.half_life)
        P[te] = RecencyEnsemble().fit(Xtr, ytr, sample_weight=w).predict_proba(Xn[te])[:, 1]
    print("Local reward (cross-validated, honest):")
    _report("same-version CV (deploy-like)", P, yn)

    # ---- cross-version holdout (pessimistic) ----
    ph = RecencyEnsemble().fit(
        oldX, oldy, sample_weight=recency_weights(old_dates, max(old_dates), half_life_days=args.half_life)
    ).predict_proba(Xn)[:, 1]
    _report("cross-version holdout (worst-case)", ph, yn)

    # ---- LODO pooled (conservative) ----
    preds: List[float] = []
    labs: List[int] = []
    for d in dates:
        tr_dates = [x for x in dates if x != d]
        Xtr = np.vstack([by_date[x][0] for x in tr_dates])
        ytr = np.concatenate([by_date[x][1] for x in tr_dates])
        row_dates = np.concatenate([[x] * len(by_date[x][1]) for x in tr_dates]).tolist()
        w = recency_weights(row_dates, d, half_life_days=args.half_life)
        Xte, yte = by_date[d]
        preds.extend(RecencyEnsemble().fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1])
        labs.extend(yte)
    _report("leave-one-date-out pooled", np.asarray(preds), np.asarray(labs))


if __name__ == "__main__":
    main()
