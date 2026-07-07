"""Train a Poker44 bot-detection model and evaluate it with the EXACT subnet reward.

Why this matters: the on-chain reward is
    reward = 0.75 * average_precision + 0.25 * recall @ (FPR <= 5%)
computed over a rolling window of chunk scores (see poker44/score/scoring.py),
and emissions are winner-take-all. So we:
  * validate on the subnet's own `reward()` (not accuracy),
  * hold out whole release DATES (never mix a date across train/val) to select
    for generalization -- the live bots evolve, and overfitting one date loses.

Usage (from repo root, with the project venv):
    python -m miner_training.train --auto --n-dates 6
    python -m miner_training.train --train-dates 2026-07-06 --val-dates 2026-07-05,2026-07-04
    python -m miner_training.train --train-dates 2026-07-06 --holdout-frac 0.25   # in-date split

Outputs a pickle: {"model", "feature_names"} for miner_training/serve.py.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from poker44.score.scoring import reward
from miner_training import benchmark_client as bc
from miner_training.features import FEATURE_NAMES, extract_features
from miner_training.ensemble import RecencyEnsemble, recency_weights


def _build_xy(
    dates: List[str], *, cache_dir: str, force: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X: List[List[float]] = []
    y: List[int] = []
    row_dates: List[str] = []
    used: List[str] = []
    for date in dates:
        records = bc.get_date(date, cache_dir=cache_dir, force=force)
        count = 0
        for hands, label, meta in bc.iter_examples(records):
            X.append(extract_features(hands))
            y.append(int(label))
            row_dates.append(str(meta.get("sourceDate") or date))
            count += 1
        if count:
            used.append(date)
        print(f"  {date}: {count} labeled groups")
    if not X:
        raise SystemExit("No training examples found for the requested dates.")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(row_dates), used


def _evaluate(model, X: np.ndarray, y: np.ndarray, label: str) -> float:
    proba = model.predict_proba(X)[:, 1]
    rew, res = reward(proba, y)
    print(
        f"  [{label}] subnet_reward={rew:.4f} | AP={res['ap_score']:.4f} "
        f"| bot_recall@fpr5%={res['bot_recall']:.4f} | fpr={res['fpr']:.4f} "
        f"| n={len(y)} pos_rate={y.mean():.3f}"
    )
    return rew


def _top_importances(model, names, k: int = 20) -> None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return
    order = np.argsort(imp)[::-1][:k]
    print(f"\nTop {k} features:")
    for i in order:
        print(f"  {names[i]:24s} {imp[i]:.4f}")


def _drift_select(X_full: np.ndarray, capture_path: str, max_drift: float):
    """Keep only features whose distribution is stable benchmark->live (low mean shift)."""
    from miner_training.gap_diagnose import _load_live

    L, _ = _load_live(capture_path)
    if len(L) == 0:
        print(f"  --robust: no live captures at {capture_path!r}; keeping ALL {len(FEATURE_NAMES)} features.")
        return list(range(len(FEATURE_NAMES))), list(FEATURE_NAMES)
    keep_idx, keep_names, dropped = [], [], []
    for i, name in enumerate(FEATURE_NAMES):
        b = X_full[:, i]
        sd = b.std() or 1.0
        shift = abs((L[:, i].mean() - b.mean()) / sd)
        if shift <= max_drift:
            keep_idx.append(i)
            keep_names.append(name)
        else:
            dropped.append((shift, name))
    dropped.sort(reverse=True)
    print(f"  --robust: kept {len(keep_idx)}/{len(FEATURE_NAMES)} stable features (|shift|<={max_drift}) "
          f"from {len(L)} live chunks; dropped {len(dropped)} drifting.")
    if dropped:
        print("    top dropped:", ", ".join(n for _, n in dropped[:8]))
    if len(keep_idx) < 5:
        print("    WARNING: very few features survived; raise --max-drift.")
    return keep_idx, keep_names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", dest="use_all", action="store_true",
                    help="download & use ALL release dates (hold out newest for the local test)")
    ap.add_argument("--fit-all", dest="fit_all", action="store_true", default=None,
                    help="after the held-out test, refit on ALL data and save that (default: on for --all)")
    ap.add_argument("--no-fit-all", dest="fit_all", action="store_false",
                    help="save the held-out-trained model instead of a refit-on-all model")
    ap.add_argument("--auto", action="store_true", help="use the most recent --n-dates release dates")
    ap.add_argument("--n-dates", type=int, default=6)
    ap.add_argument("--train-dates", default="", help="comma list YYYY-MM-DD")
    ap.add_argument("--val-dates", default="", help="comma list YYYY-MM-DD (held out)")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="if no --val-dates, stratified in-date holdout fraction")
    ap.add_argument("--cache-dir", default=bc.DEFAULT_CACHE)
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--out", default="miner_training/model.pkl")
    ap.add_argument("--half-life", type=float, default=10.0,
                    help="recency half-life in days (older release dates get exponentially less weight)")
    ap.add_argument("--robust", action="store_true",
                    help="drift-aware: use captured live queries to DROP features that don't transfer to live")
    ap.add_argument("--capture", default="live_capture/live_chunks.jsonl",
                    help="captured live queries for --robust drift selection")
    ap.add_argument("--max-drift", type=float, default=1.0,
                    help="with --robust, keep features whose |mean shift| vs live is <= this")
    args = ap.parse_args()

    if args.use_all:
        print("Downloading ALL release dates...")
        dates = bc.download_all(cache_dir=args.cache_dir, force=args.force_download)
        if len(dates) < 2:
            raise SystemExit("Need at least 2 dates for a held-out test.")
        # newest date is the most representative of live bots -> hold it out for the local test
        val_dates = [dates[-1]]
        train_dates = dates[:-1]
    elif args.auto or not args.train_dates:
        dates = bc.recent_dates(args.n_dates)
        if not dates:
            raise SystemExit("Could not discover release dates from the API.")
        # newest date held out for validation, the rest for training
        val_dates = [dates[-1]]
        train_dates = dates[:-1] or dates
    else:
        train_dates = [d.strip() for d in args.train_dates.split(",") if d.strip()]
        val_dates = [d.strip() for d in args.val_dates.split(",") if d.strip()]

    print(f"Feature dim: {len(FEATURE_NAMES)}")
    print(f"Train dates: {train_dates}")
    print(f"Val dates:   {val_dates or '(in-date holdout)'}\n")

    print("Loading TRAIN:")
    X, y, xdates, _ = _build_xy(train_dates, cache_dir=args.cache_dir, force=args.force_download)

    Xval: Optional[np.ndarray] = None
    yval: Optional[np.ndarray] = None
    if val_dates:
        print("Loading VAL:")
        Xval, yval, xvaldates, _ = _build_xy(val_dates, cache_dir=args.cache_dir, force=args.force_download)
    elif args.holdout_frac > 0:
        from sklearn.model_selection import train_test_split

        X, Xval, y, yval, xdates, xvaldates = train_test_split(
            X, y, xdates, test_size=args.holdout_frac, stratify=y, random_state=0
        )

    # Drift-aware feature selection: drop features that don't transfer benchmark->live.
    selected_names = list(FEATURE_NAMES)
    if args.robust:
        print("\nDrift-aware feature selection (--robust):")
        sel_idx, selected_names = _drift_select(X, args.capture, args.max_drift)
        X = X[:, sel_idx]
        if Xval is not None:
            Xval = Xval[:, sel_idx]

    # Recency-weight the training rows: recent release dates count more (bots evolve).
    ref_train = max(xdates.tolist())
    w = recency_weights(xdates.tolist(), ref_train, half_life_days=args.half_life)
    model = RecencyEnsemble()
    print(
        f"\nModel: RecencyEnsemble ({len(model.estimators)} estimators) "
        f"| train n={len(y)} pos_rate={y.mean():.3f} | half_life={args.half_life}d"
    )
    model.fit(X, y, sample_weight=w)

    print("\nEvaluation:")
    _evaluate(model, X, y, "train")
    held_out_reward = None
    if Xval is not None and yval is not None and len(yval):
        held_out_reward = _evaluate(model, Xval, yval, "val (held-out)")
    _top_importances(model, selected_names)

    # Decide which model to persist. In --all mode (or with --fit-all) we refit a
    # fresh model on train+val so the deployed model uses every labeled example;
    # the held-out number above is the honest local estimate of live performance.
    fit_all = args.fit_all if args.fit_all is not None else bool(args.use_all)
    save_model = model
    if fit_all and Xval is not None and yval is not None and len(yval):
        Xfull = np.vstack([X, Xval])
        yfull = np.concatenate([y, yval])
        dates_full = np.concatenate([xdates, xvaldates]).tolist()
        wf = recency_weights(dates_full, max(dates_full), half_life_days=args.half_life)
        deploy = RecencyEnsemble()
        deploy.fit(Xfull, yfull, sample_weight=wf)
        save_model = deploy
        print(
            f"\nRefit deploy model on ALL data (n={len(yfull)}, pos_rate={yfull.mean():.3f}). "
            + (f"Local held-out estimate: reward={held_out_reward:.4f}." if held_out_reward is not None else "")
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump({"model": save_model, "feature_names": selected_names}, fh)
    print(f"Saved model -> {out_path}")


if __name__ == "__main__":
    main()
