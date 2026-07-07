"""Capture the UNLABELED live validator queries for benchmark->live drift analysis.

The public benchmark is a misleading proxy for the live eval (our model read 0.74
on the benchmark but ~0.49 live). Live queries carry the REAL distribution, so
logging them lets us measure which features drift benchmark->live and stop
overfitting the benchmark. Live queries have NO bot/human label — this is for
domain-adaptation / diagnosis only, never supervised labels.

Safety contract:
  * OFF by default. Enable with env POKER44_CAPTURE=1.
  * FAIL-SAFE: every path is wrapped; a capture error can never affect serving/scoring.
  * Deduped by chunk content hash (validators resend the same daily snapshot).
  * Size-capped (POKER44_CAPTURE_MAX_BYTES, default 200MB). Output is gitignored.

INTEGRITY: captures are unlabeled, so nothing here is a training label and your
training-data statement is unchanged while you only DIAGNOSE with them. If you
ever feed captures into training (even unlabeled, for domain adaptation), update
POKER44_MODEL_PRIVATE_DATA_ATTESTATION truthfully.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

_LOCK = threading.Lock()
_STATE = {"path": None, "seen": None, "full": False, "bytes": 0}


def _enabled() -> bool:
    return os.getenv("POKER44_CAPTURE", "").strip().lower() in {"1", "true", "yes", "on"}


def _capture_dir() -> Path:
    return Path(os.getenv("POKER44_CAPTURE_DIR") or "live_capture")


def _max_bytes() -> int:
    try:
        return int(os.getenv("POKER44_CAPTURE_MAX_BYTES", str(200 * 1024 * 1024)))
    except ValueError:
        return 200 * 1024 * 1024


def _chunk_hash(chunk: Sequence[dict]) -> str:
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()


def _init_locked() -> None:
    d = _capture_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "live_chunks.jsonl"
    seen: set = set()
    total = 0
    if path.exists():
        total = path.stat().st_size
        try:
            with path.open() as fh:
                for line in fh:
                    try:
                        seen.add(json.loads(line)["chunk_hash"])
                    except Exception:
                        continue
        except Exception:
            pass
    _STATE.update(path=path, seen=seen, bytes=total, full=total >= _max_bytes())


def capture_chunks(chunks: List[List[dict]], scores: Optional[List[float]] = None) -> None:
    """Append any new (deduped) live chunks. No-op unless POKER44_CAPTURE=1. Never raises."""
    try:
        if not _enabled() or not chunks:
            return
        with _LOCK:
            if _STATE["seen"] is None:
                _init_locked()
            if _STATE["full"]:
                return
            rows: List[str] = []
            for i, chunk in enumerate(chunks):
                try:
                    h = _chunk_hash(chunk)
                    if h in _STATE["seen"]:
                        continue
                    _STATE["seen"].add(h)
                    rows.append(json.dumps(
                        {
                            "chunk_hash": h,
                            "ts": time.time(),
                            "n_hands": len(chunk),
                            "score": (float(scores[i]) if scores and i < len(scores) else None),
                            "chunk": chunk,
                        },
                        separators=(",", ":"),
                        default=str,
                    ))
                except Exception:
                    continue
            if rows:
                blob = "\n".join(rows) + "\n"
                with _STATE["path"].open("a") as fh:
                    fh.write(blob)
                _STATE["bytes"] += len(blob.encode("utf-8", "ignore"))
                if _STATE["bytes"] >= _max_bytes():
                    _STATE["full"] = True
    except Exception:
        return  # fail-safe: capture must never affect serving
