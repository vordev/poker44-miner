"""Runtime metadata helpers for validator observability."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def collect_runtime_info() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    short_commit = _run_git("rev-parse", "--short", "HEAD")
    branch = _run_git("branch", "--show-current")
    dirty = bool(_run_git("status", "--porcelain"))
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": commit,
        "git_commit_short": short_commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "pid": os.getpid(),
        "started_at": time.time(),
    }


def write_runtime_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def post_runtime_snapshot(
    *,
    url: str,
    hotkey_ss58: str,
    signature_hex: str,
    nonce: str,
    timestamp: int,
    payload: Mapping[str, Any],
    timeout_seconds: float = 5.0,
) -> tuple[bool, str]:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-validator-hotkey": hotkey_ss58,
            "x-validator-signature": signature_hex,
            "x-validator-nonce": nonce,
            "x-validator-timestamp": str(timestamp),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            body = response.read().decode("utf-8", errors="replace")
        return 200 <= int(status) < 300, body[:500]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return False, f"http_{exc.code}:{body[:500]}"
    except Exception as exc:
        return False, str(exc)


def build_signed_runtime_request(
    *,
    wallet: Any,
    url: str,
    payload: Optional[Mapping[str, Any]],
    method: str = "POST",
) -> dict[str, Any]:
    normalized_method = method.upper()
    if payload is None and normalized_method in {"GET", "HEAD"}:
        encoded = b""
    else:
        encoded = json.dumps(payload or {}, sort_keys=True).encode("utf-8")
    parsed = urlparse(url)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    nonce = secrets.token_hex(16)
    timestamp = int(time.time())
    body_hash = sha256(encoded).hexdigest()
    message = f"{timestamp}:{nonce}:{normalized_method}:{path}:{body_hash}"
    signature_hex = wallet.hotkey.sign(message.encode()).hex()
    return {
        "hotkey_ss58": wallet.hotkey.ss58_address,
        "signature_hex": signature_hex,
        "nonce": nonce,
        "timestamp": timestamp,
    }
