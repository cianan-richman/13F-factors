"""SEC EDGAR HTTP client with thread-safe rate limiting, connection pooling, and retries."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

_PLACEHOLDER_AGENTS = {
    "",
    "your name your.email@example.com",
    "your name your.email@domain.com",
    "name email",
}
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class EdgarConfigError(ValueError):
    """Raised when required SEC EDGAR configuration (such as User-Agent) is missing."""


class EdgarBlockedError(requests.HTTPError):
    """Raised on HTTP 403 from SEC EDGAR (missing/rejected User-Agent or temporary IP block)."""


class EdgarClient:
    """Thread-safe, rate-limited HTTP client compliant with SEC EDGAR fair-access guidelines."""

    def __init__(
        self,
        user_agent: str,
        max_requests_per_sec: float = 8.0,
        max_retries: int = 5,
        backoff_base_sec: float = 1.0,
        connect_timeout_sec: float = 10.0,
        read_timeout_sec: float = 30.0,
        pool_maxsize: int = 10,
        session: requests.Session | None = None,
    ) -> None:
        cleaned_agent = (user_agent or "").strip()
        if cleaned_agent.lower() in _PLACEHOLDER_AGENTS or "@" not in cleaned_agent:
            raise EdgarConfigError(
                "Invalid or missing SEC_USER_AGENT. Set SEC_USER_AGENT='FirstName LastName "
                "email@domain.com' in your .env file per SEC EDGAR fair-access policy."
            )
        if max_requests_per_sec <= 0 or max_requests_per_sec > 10:
            raise ValueError("max_requests_per_sec must be in (0, 10] per SEC EDGAR limits.")

        self.user_agent = cleaned_agent
        self.min_interval_sec = 1.0 / max_requests_per_sec
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.timeout: tuple[float, float] = (connect_timeout_sec, read_timeout_sec)

        self._lock = threading.Lock()
        self._next_allowed_ts: float = 0.0

        self._session = session or requests.Session()
        if session is None:
            adapter = HTTPAdapter(
                pool_connections=pool_maxsize,
                pool_maxsize=pool_maxsize,
                max_retries=0,
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)

        self._session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def _acquire_slot(self) -> None:
        """Reserve the next rate-limited request slot safely across worker threads."""
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed_ts)
            self._next_allowed_ts = scheduled + self.min_interval_sec
        wait_sec = scheduled - now
        if wait_sec > 0:
            time.sleep(wait_sec)

    def get(self, url: str, *, allow_not_found: bool = False) -> requests.Response:
        """Issue a throttled GET request with exponential backoff on retryable errors."""
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._acquire_slot()
            try:
                resp = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                sleep_sec = self.backoff_base_sec * (2**attempt)
                logger.warning(
                    "EDGAR request error (%s) on %s; retrying in %.1fs (attempt %d/%d)",
                    exc,
                    url,
                    sleep_sec,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(sleep_sec)
                continue

            if resp.status_code == 200:
                return resp

            if resp.status_code == 404 and allow_not_found:
                return resp

            if resp.status_code == 403:
                raise EdgarBlockedError(
                    f"SEC EDGAR returned HTTP 403 Forbidden on {url}. Verify SEC_USER_AGENT "
                    "('FirstName LastName email@domain.com') and wait before retrying if rate-"
                    "limited.",
                    response=resp,
                )

            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_sec = max(float(retry_after), self.backoff_base_sec)
                else:
                    sleep_sec = self.backoff_base_sec * (2**attempt)
                logger.warning(
                    "EDGAR HTTP %d on %s; retrying in %.1fs (attempt %d/%d)",
                    resp.status_code,
                    url,
                    sleep_sec,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(sleep_sec)
                continue

            resp.raise_for_status()

        assert last_exc is not None
        raise last_exc
