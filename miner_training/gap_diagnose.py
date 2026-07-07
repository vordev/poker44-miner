"""Measure the benchmark -> live gap from captured live queries.

Run AFTER the miner has captured live queries (POKER44_CAPTURE=1, then wait for
validators to hit you). It shows two things:

  1. MODEL GAP  - your model's score distribution on the benchmark vs on the live
     captures. If it spreads on the benchmark but collapses near a constant live,
     the model isn't discriminating live (that's the 0.74->0.49 gap made visible).
  2. FEATURE DRIFT - per-feature standardized mean shift between benchmark and
     live. High-shift features behave differently live, so a model leaning on them
     (our sizing features) won't transfer. Rebuild on the low-shift, stable ones.

The live eval is tiny (~21 unique chunks/day), so this uses standardized mean
shift (robust at small n) as the primary signal; PSI is shown only as a rough
secondary and is unreliable below ~30 live chunks.

    python -m miner_training.gap_diagnose
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from miner_training import benchmark_client as bc
from miner_training.features import FEATURE_NAMES, extract_features


def _load_benchmark(cache_dir: str) -> np.ndarray:
    root = Path(cache_dir)
    dates = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    X = []
    for d in dates:
        for hands, _label, _meta in bc.iter_examples(bc.load_cached_date(d, cache_dir)):
            X.append(extract_features(hands))
    return np.asarray(X, dtype=float)


def _load_live(capture_path: str):
    X, scores = [], []
    p = Path(capture_path)
    if not p.exists():
        return np.zeros((0, len(FEATURE_NAMES))), []
    with p.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                X.append(extract_features(row["chunk"]))
                if row.get("score") is not None:
                    scores.append(float(row["score"]))
            except Exception:
                continue
    return np.asarray(X, dtype=float), scores


def _psi(bench: np.ndarray, live: np.ndarray, bins: int = 5) -> float:
    edges = np.unique(np.quantile(bench, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    b, _ = np.histogram(bench, bins=edges)
    l, _ = np.histogram(live, bins=edges)
    eps = 1e-6
    bf = np.clip(b / max(1, b.sum()), eps, None)
    lf = np.clip(l / max(1, l.sum()), eps, None)
    return float(np.sum((lf - bf) * np.log(lf / bf)))


def _dist(x: np.ndarray) -> str:
    return f"mean={x.mean():.3f} std={x.std():.3f} p10={np.quantile(x,0.1):.3f} p90={np.quantile(x,0.9):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default=bc.DEFAULT_CACHE)
    ap.add_argument("--capture", default="live_capture/live_chunks.jsonl")
    ap.add_argument("--model", default="miner_training/model.pkl")
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    B = _load_benchmark(args.cache_dir)
    L, live_scores = _load_live(args.capture)

    if len(L) == 0:
        raise SystemExit(
            f"No live captures at {args.capture!r}.\n"
            "1) relaunch the miner with POKER44_CAPTURE=1\n"
            "2) wait for validators to query you (watch: pm2 logs poker44_miner | grep Scored)\n"
            "3) re-run this."
        )
    if len(B) == 0:
        raise SystemExit("No benchmark cache; run `python -m miner_training.train --all` first.")

    print(f"benchmark chunks={len(B)}   live chunks captured={len(L)}")
    if len(L) < 30:
        print(f"NOTE: only {len(L)} live chunks — trust the mean-shift ranking, ignore PSI until you have ~30+.\n")

    # ---- 1. MODEL GAP: run the CURRENT model on benchmark vs live captures ----
    mp = Path(args.model)
    if mp.exists():
        blob = pickle.load(mp.open("rb"))
        model = blob["model"]
        names = blob.get("feature_names", FEATURE_NAMES)
        idx = [FEATURE_NAMES.index(n) for n in names]  # model may use a feature SUBSET
        pB = model.predict_proba(B[:, idx])[:, 1]
        pL = model.predict_proba(L[:, idx])[:, 1]
        collapsed = pL.std() < 0.10
        print(f"CURRENT model ({len(names)} features) score distribution:")
        print(f"  benchmark: {_dist(pB)}")
        print(f"  live     : {_dist(pL)}"
              + ("  <-- STILL COLLAPSED (not discriminating live)" if collapsed else "  <-- DISCRIMINATING live"))
        if live_scores:
            s = np.asarray(live_scores)
            print(f"  (as served when captured: mean={s.mean():.3f} std={s.std():.3f})")
        print()

    # ---- 2. FEATURE DRIFT: standardized mean shift (primary), PSI (secondary) ----
    rows = []
    for i, name in enumerate(FEATURE_NAMES):
        b, l = B[:, i], L[:, i]
        sd = b.std() or 1.0
        rows.append((abs((l.mean() - b.mean()) / sd), _psi(b, l), name, b.mean(), l.mean()))
    rows.sort(reverse=True)

    print("TOP DRIFT features by |mean shift| (benchmark != live -> model overfits these, drop/down-weight):")
    print(f"  {'feature':28} {'mean_shift':>10} {'PSI~':>6} {'bench':>9} {'live':>9}")
    for sh, psi, name, bm, lm in rows[: args.top]:
        flag = "  <-- big" if sh > 1.0 else ("  <- moderate" if sh > 0.5 else "")
        print(f"  {name:28} {sh:10.2f} {psi:6.2f} {bm:9.3f} {lm:9.3f}{flag}")

    print("\nMOST STABLE features (smallest shift -- keep/build the new model on these):")
    for sh, psi, name, bm, lm in sorted(rows)[:12]:
        print(f"  {name:28} shift={sh:.3f}")

    big = sum(1 for r in rows if r[0] > 1.0)
    print(f"\nSummary: {big}/{len(rows)} features shift hard (|mean shift|>1 std). "
          "Retrain excluding the high-shift set, redeploy, and re-check your live round score.")


if __name__ == "__main__":
    main()
