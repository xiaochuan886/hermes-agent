"""Authenticated sync client for the SkillHub control plane.

Fetches the device's ``RuntimeAssignment`` set from
``GET /api/v1/me/agent-runtime`` with:

* ``Authorization: Bearer <token>`` — the token is injected by the caller; this
  module never performs a login flow.
* a stable ``X-Device-Id`` header identifying the installation.
* ``If-None-Match`` conditional requests driven by the last received ETag.

Retry policy: bounded exponential backoff on transport timeouts and 5xx
responses; 401/403 are authentication failures and are never retried; other 4xx
are surfaced immediately.

Logging discipline: request/response logs contain only the method, URL path,
status code, ETag, retry count and per-assignment metadata (id, revision,
revoked, slug). The Authorization header, the response body (which carries the
manifest), and any provider credentials are never logged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import httpx

from hermes_managed.contracts import (
    RuntimeAssignment,
    parse_runtime_assignment,
)

__all__ = [
    "ControlPlaneError",
    "AuthenticationError",
    "TransientFailureError",
    "UnexpectedStatusError",
    "SyncResult",
    "ControlPlaneClient",
]

_LOGGER = logging.getLogger("hermes_managed.control_plane")

_AGENT_RUNTIME_PATH = "/api/v1/me/agent-runtime"


# --- exceptions ---------------------------------------------------------------


class ControlPlaneError(Exception):
    """Base class for control-plane sync failures."""


class AuthenticationError(ControlPlaneError):
    """401/403 — the injected token was rejected; do not retry."""


class TransientFailureError(ControlPlaneError):
    """Timeout or 5xx persisted after all retries."""


class UnexpectedStatusError(ControlPlaneError):
    """A non-retryable, non-auth HTTP status was returned."""


# --- result -------------------------------------------------------------------


@dataclass(frozen=True)
class SyncResult:
    """Outcome of a sync attempt.

    ``not_modified`` is True when the server returned 304. ``etag`` is the
    server-reported ETag (use it as the next ``If-None-Match`` value).
    """

    assignments: tuple[RuntimeAssignment, ...]
    etag: Optional[str]
    not_modified: bool
    status_code: int


# --- client -------------------------------------------------------------------


class ControlPlaneClient:
    """Syncs runtime assignments from the SkillHub control plane."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        device_id: str,
        http_client: Optional[httpx.Client] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
        retry_backoff_factor: float = 2.0,
        retry_after_max: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not token:
            raise ValueError("token must be a non-empty string")
        if not device_id:
            raise ValueError("device_id must be a non-empty string")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self._base_url = base_url.rstrip("/")
        self._token = token
        self._device_id = device_id
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_backoff_factor = retry_backoff_factor
        self._retry_after_max = retry_after_max
        self._sleep = sleep
        self._logger = logger or _LOGGER

        self._owns_client = http_client is None
        self._http: httpx.Client = http_client or httpx.Client(timeout=timeout)

    # -- public API --

    def fetch_runtime_assignments(self, *, etag: Optional[str] = None) -> SyncResult:
        """Fetch the current assignment set, conditionally if ``etag`` is given."""
        url = self._base_url + _AGENT_RUNTIME_PATH
        headers = self._request_headers(etag)
        self._logger.debug("control plane request: GET %s", _AGENT_RUNTIME_PATH)

        last_error: Optional[BaseException] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.get(url, headers=headers, timeout=self._timeout)
            except httpx.TransportError as exc:
                # Timeouts, connection errors, protocol errors — retryable.
                # (httpx.TimeoutException is a subclass of TransportError.)
                last_error = exc
                if attempt < self._max_retries:
                    self._sleep(self._backoff_delay(attempt))
                    self._logger.warning(
                        "control plane transport error %s (attempt %d/%d); retrying",
                        type(exc).__name__, attempt + 1, self._max_retries + 1,
                    )
                    continue
                raise TransientFailureError(
                    f"control plane transport error after {attempt + 1} attempts: "
                    f"{type(exc).__name__}"
                ) from exc

            status = response.status_code
            if status == 200:
                return self._handle_ok(response)
            if status == 304:
                return self._handle_not_modified(response, etag)
            if status in (401, 403):
                # Authentication/authorization failure — never retry.
                raise AuthenticationError(f"control plane rejected credentials: {status}")
            if status == 429:
                # Throttled: retry only if the server asks us to (Retry-After),
                # capped; otherwise surface immediately.
                retry_after = self._parse_retry_after(response)
                if retry_after is not None and attempt < self._max_retries:
                    self._sleep(retry_after)
                    self._logger.warning(
                        "control plane throttled (429); retrying after %.3fs",
                        retry_after,
                    )
                    continue
                raise UnexpectedStatusError("control plane throttled (429) without retry")
            if 500 <= status < 600:
                last_error = None
                if attempt < self._max_retries:
                    delay = self._parse_retry_after(response)
                    if delay is None:
                        delay = self._backoff_delay(attempt)
                    self._sleep(delay)
                    self._logger.warning(
                        "control plane returned %d (attempt %d/%d); retrying",
                        status, attempt + 1, self._max_retries + 1,
                    )
                    continue
                raise TransientFailureError(
                    f"control plane returned {status} after {attempt + 1} attempts"
                )
            # Any other 4xx is a contract/client error — do not retry.
            raise UnexpectedStatusError(f"control plane returned unexpected status: {status}")

        # Unreachable: the loop either returns or raises.
        raise TransientFailureError(  # pragma: no cover
            f"control plane sync failed: {last_error}"
        )

    def close(self) -> None:
        """Close the underlying HTTP client if this wrapper owns it."""
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "ControlPlaneClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals --

    def _request_headers(self, etag: Optional[str]) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-Device-Id": self._device_id,
            "Accept": "application/json",
        }
        if etag:
            headers["If-None-Match"] = etag
        return headers

    def _backoff_delay(self, attempt: int) -> float:
        return self._retry_base_delay * (self._retry_backoff_factor ** attempt)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """Return a capped sleep duration (seconds) from a ``Retry-After`` header.

        Only the integer-seconds form is honored; an HTTP-date value (or a
        missing header) returns ``None`` so the caller falls back to exponential
        backoff.  The duration is capped at ``retry_after_max``.
        """
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if seconds < 0:
            return None
        return min(seconds, self._retry_after_max)

    def _handle_ok(self, response: httpx.Response) -> SyncResult:
        etag = response.headers.get("ETag")
        try:
            body = response.json()
        except Exception as exc:  # malformed JSON
            raise ControlPlaneError("control plane returned non-JSON body") from exc
        if not isinstance(body, dict) or "assignments" not in body:
            raise ControlPlaneError("control plane response missing 'assignments'")
        raw_assignments = body["assignments"]
        if not isinstance(raw_assignments, list):
            raise ControlPlaneError("control plane 'assignments' is not a list")

        assignments: list[RuntimeAssignment] = []
        for item in raw_assignments:
            # Fail closed: a single malformed assignment aborts the sync.
            assignments.append(parse_runtime_assignment(item))
        self._log_assignments(assignments)
        return SyncResult(
            assignments=tuple(assignments),
            etag=etag,
            not_modified=False,
            status_code=200,
        )

    def _handle_not_modified(self, response: httpx.Response, etag: Optional[str]) -> SyncResult:
        server_etag = response.headers.get("ETag") or etag
        self._logger.debug("control plane: not modified (304)")
        return SyncResult(
            assignments=(),
            etag=server_etag,
            not_modified=True,
            status_code=304,
        )

    def _log_assignments(self, assignments: list[RuntimeAssignment]) -> None:
        # Log only non-sensitive metadata — never the manifest, policy, or creds.
        for a in assignments:
            self._logger.debug(
                "assignment received: id=%s revision=%s version_id=%s revoked=%s slug=%s",
                a.assignment_id, a.revision, a.version_id, a.revoked, a.template_slug,
            )
        self._logger.info("control plane sync ok: %d assignment(s)", len(assignments))
