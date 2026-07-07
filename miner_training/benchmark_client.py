"""Client for the public Poker44 training-benchmark API.

Base: https://api.poker44.net/api/v1/benchmark  (public, no auth)

Endpoints used:
    GET /                                   -> status (latestSourceDate, ...)
    GET /releases?limit=N                   -> release history
    GET /chunks?sourceDate=YYYY-MM-DD&...   -> chunk records (paginated)

Each chunk *record* bundles several labeled *groups*:
    record["chunks"]      -> List[group], group = List[hand dict]  (miner input)
    record["groundTruth"] -> List[int],   1 = bot, 0 = human       (aligned by index)

We cache each record to disk by `chunkHash` so experiments are reproducible
(as the benchmark docs recommend).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

BASE_URL = os.getenv("POKER44_BENCHMARK_URL", "https://api.poker44.net/api/v1/benchmark").rstrip("/")
DEFAULT_CACHE = os.getenv("POKER44_BENCHMARK_CACHE", "benchmark_cache")
_MAX_ATTEMPTS = 5


def _get(path: str = "", params: Optional[Dict[str, Any]] = None, timeout: float = 60.0) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(BASE_URL + path, params=params or {}, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("success") is False:
                raise RuntimeError(f"benchmark API error: {payload}")
            return payload.get("data", payload) if isinstance(payload, dict) else payload
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(min(2.0 * attempt, 8.0))  # backoff on transient DNS/network errors
    raise RuntimeError(f"benchmark request failed after {_MAX_ATTEMPTS} attempts: {path} -> {last_exc}")


def get_status() -> Dict[str, Any]:
    """Return availability info; `latestSourceDate` is the source of truth."""
    return _get("")


def get_releases(limit: int = 30, before: Optional[str] = None) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": limit}
    if before:
        params["before"] = before
    data = _get("/releases", params)
    if isinstance(data, dict):
        return data.get("releases", [])
    return data or []


def latest_source_date() -> str:
    return str(get_status()["latestSourceDate"])


def recent_dates(n: int = 4) -> List[str]:
    """Newest-last list of the n most recent release dates."""
    rels = get_releases(limit=max(n, 1))
    dates = sorted({str(r["sourceDate"]) for r in rels if r.get("sourceDate")})
    return dates[-n:]


def all_dates() -> List[str]:
    """Every release date, oldest-first, via full /releases pagination."""
    seen: List[str] = []
    cursor: Optional[str] = None
    for _ in range(200):  # hard stop; there are ~dozens of dates
        page = get_releases(limit=100, before=cursor)
        dates = [str(r["sourceDate"]) for r in page if r.get("sourceDate")]
        fresh = [d for d in dates if d not in seen]
        if not fresh:
            break
        seen.extend(fresh)
        cursor = min(dates)  # step older
        if len(page) < 100:
            break
    return sorted(set(seen))


def download_all(
    *,
    cache_dir: str = DEFAULT_CACHE,
    force: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Download + cache every release date. Returns dates fetched (oldest-first)."""
    dates = all_dates()
    if verbose:
        print(f"Discovered {len(dates)} release dates: {dates[0]} .. {dates[-1]}")
    total_groups = 0
    for i, date in enumerate(dates, 1):
        records = get_date(date, cache_dir=cache_dir, force=force)
        groups = sum(len(r.get("groundTruth") or []) for r in records)
        total_groups += groups
        if verbose:
            print(f"  [{i:>2}/{len(dates)}] {date}: {len(records):>3} records, {groups:>4} groups")
    if verbose:
        print(f"Total labeled groups across all dates: {total_groups}")
    return dates


def _date_dir(source_date: str, cache_dir: str) -> Path:
    return Path(cache_dir) / source_date


def download_date(
    source_date: str,
    *,
    split: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE,
    page_limit: int = 24,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Download every chunk record for one release date (paginated) and cache it."""
    out_dir = _date_dir(source_date, cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"sourceDate": source_date, "limit": page_limit}
        if split:
            params["split"] = split
        if cursor:
            params["cursor"] = cursor
        data = _get("/chunks", params)
        page = data.get("chunks", []) if isinstance(data, dict) else []
        for rec in page:
            key = str(rec.get("chunkHash") or rec.get("chunkId") or len(records))
            (out_dir / f"{key}.json").write_text(json.dumps(rec))
            records.append(rec)
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if max_records is not None and len(records) >= max_records:
            break
        if not cursor:
            break
    return records


def load_cached_date(source_date: str, cache_dir: str = DEFAULT_CACHE) -> List[Dict[str, Any]]:
    out_dir = _date_dir(source_date, cache_dir)
    if not out_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def get_date(
    source_date: str,
    *,
    split: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE,
    force: bool = False,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return records for a date, using the on-disk cache unless `force`."""
    if not force:
        cached = load_cached_date(source_date, cache_dir)
        if cached:
            return cached
    return download_date(source_date, split=split, cache_dir=cache_dir, max_records=max_records)


def iter_examples(
    records: List[Dict[str, Any]],
) -> Iterator[Tuple[List[dict], int, Dict[str, Any]]]:
    """Yield (hands, label, meta) for every labeled group across the records.

    label: 1 = bot, 0 = human. meta carries provenance (never use it as a feature).
    """
    for rec in records:
        groups = rec.get("chunks") or []
        labels = rec.get("groundTruth") or []
        source_date = rec.get("sourceDate")
        for idx, (group, label) in enumerate(zip(groups, labels)):
            yield group, int(label), {
                "chunkId": rec.get("chunkId"),
                "chunkHash": rec.get("chunkHash"),
                "sourceDate": source_date,
                "releaseVersion": rec.get("releaseVersion"),
                "split": rec.get("split"),
                "group_index": idx,
            }


if __name__ == "__main__":  # tiny smoke CLI: python -m miner_training.benchmark_client
    status = get_status()
    print("latestSourceDate:", status.get("latestSourceDate"), "| totalHands:", status.get("totalHands"))
    for r in get_releases(limit=8):
        print(" ", r.get("sourceDate"), "chunks", r.get("chunkCount"), "hands", r.get("handCount"))
