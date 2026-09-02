"""Shared HTTP plumbing: one polite, retrying session for every non-nflreadpy fetch.

nflreadpy manages its own session for nflverse assets. This module is for the
supplementary sources (weather, coverage probes) and for HEAD-probing nflverse
release assets without downloading them.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .config import get_settings

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@lru_cache(maxsize=1)
def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": get_settings().user_agent})
    adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def request(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Issue a request with exponential backoff on transient failures.

    Honours ``Retry-After`` when the server sends it, which matters for the
    public weather APIs.
    """
    cfg = get_settings()
    kwargs.setdefault("timeout", cfg.http_timeout)
    last_exc: Exception | None = None

    for attempt in range(cfg.http_retries):
        try:
            resp = session().request(method, url, **kwargs)
            if resp.status_code in RETRYABLE_STATUS:
                wait = float(resp.headers.get("Retry-After", cfg.http_backoff**attempt))
                log.warning("%s %s -> %s; retrying in %.1fs", method, url, resp.status_code, wait)
                time.sleep(min(wait, 60))
                continue
            return resp
        except requests.RequestException as exc:  # network-level failure
            last_exc = exc
            wait = cfg.http_backoff**attempt
            log.warning("%s %s failed (%s); retrying in %.1fs", method, url, exc, wait)
            time.sleep(wait)

    if last_exc is not None:
        raise ConnectionError(f"{method} {url} failed after {cfg.http_retries} attempts") from last_exc
    raise ConnectionError(f"{method} {url} kept returning a retryable status")


def head_ok(url: str) -> bool:
    """True if the URL resolves to a real asset. Used for coverage probing."""
    try:
        resp = request("HEAD", url, allow_redirects=True)
    except ConnectionError:
        return False
    return resp.status_code == 200


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    resp = request("GET", url, params=params)
    resp.raise_for_status()
    return resp.json()
