"""Tests for hermes_managed.control_plane_client.

The client syncs ``RuntimeAssignment`` data from the SkillHub control plane.
These tests pin the frozen contract decisions: Authorization + stable
``X-Device-Id`` headers, ``If-None-Match`` / ETag handling, bounded retry on
timeout and 5xx, no retry on 401/403, and redaction of token, manifest content
and provider credentials from logs.
"""

import logging

import httpx
import pytest

from hermes_managed.contracts import canonical_json_bytes, sha256_hex
from hermes_managed.control_plane_client import (
    AuthenticationError,
    ControlPlaneError,
    ControlPlaneClient,
    SyncResult,
    TransientFailureError,
    UnexpectedStatusError,
)

TOKEN = "tk_live_SECRET_CANARY_77"
MANIFEST_CANARY = "CP_MANIFEST_CANARY_51c"


def _assignment_payload(revision: int = 2, version_id: int = 19) -> dict:
    manifest = {
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "version": "1.0.0",
        "note": MANIFEST_CANARY,
    }
    policy = {
        "mode": "ENTERPRISE_MANAGED",
        "allowedModels": ["enterprise/deepseek-chat"],
        "defaultModel": "enterprise/deepseek-chat",
        "fallbackModels": [],
        "localProviderAllowed": False,
        "policyVersion": "v1",
    }
    policy["policySha256"] = sha256_hex(canonical_json_bytes(policy))
    return {
        "assignmentId": 20,
        "revision": revision,
        "templateId": 10,
        "versionId": version_id,
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "manifest": manifest,
        "manifestSha256": sha256_hex(canonical_json_bytes(manifest)),
        "effectiveModelPolicy": policy,
        "revoked": False,
        "revokedAt": None,
    }


def _ok_response(assignments=None, etag='"v1"') -> httpx.Response:
    return httpx.Response(
        200,
        json={"assignments": assignments if assignments is not None else [_assignment_payload()]},
        headers={"ETag": etag},
    )


def _make_client(handler, *, max_retries=3, sleep=None, base_url="https://cp.example.com",
                 token=TOKEN, device_id="device-001", **kw) -> ControlPlaneClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return ControlPlaneClient(
        base_url=base_url,
        token=token,
        device_id=device_id,
        http_client=http_client,
        max_retries=max_retries,
        sleep=sleep if sleep is not None else (lambda _s: None),
        **kw,
    )


class TestHeaders:
    def test_sends_authorization_bearer_and_device_id(self):
        captured = {}

        def handler(request):
            captured["headers"] = request.headers
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return _ok_response()

        client = _make_client(handler)
        result = client.fetch_runtime_assignments()

        assert captured["method"] == "GET"
        assert captured["url"].endswith("/api/v1/me/agent-runtime")
        assert "tk_live" not in captured["url"]  # token never in URL
        assert captured["headers"]["authorization"] == f"Bearer {TOKEN}"
        assert captured["headers"]["x-device-id"] == "device-001"
        assert captured["headers"]["accept"] == "application/json"
        assert isinstance(result, SyncResult)
        assert result.status_code == 200

    def test_device_id_is_stable_across_requests(self):
        seen = []

        def handler(request):
            seen.append(request.headers["x-device-id"])
            return _ok_response()

        client = _make_client(handler)
        client.fetch_runtime_assignments()
        client.fetch_runtime_assignments()
        assert seen == ["device-001", "device-001"]

    def test_sends_if_none_match_when_etag_provided(self):
        captured = {}

        def handler(request):
            captured["if_none_match"] = request.headers.get("if-none-match")
            return httpx.Response(304, headers={"ETag": '"v1"'})

        client = _make_client(handler)
        result = client.fetch_runtime_assignments(etag='"v1"')
        assert captured["if_none_match"] == '"v1"'
        assert result.not_modified is True

    def test_omits_if_none_match_when_no_etag(self):
        captured = {}

        def handler(request):
            captured["if_none_match"] = request.headers.get("if-none-match")
            return _ok_response()

        client = _make_client(handler)
        client.fetch_runtime_assignments()
        assert captured["if_none_match"] is None

    def test_base_url_trailing_slash_normalized(self):
        captured = {}

        def handler(request):
            captured["url"] = str(request.url)
            return _ok_response()

        client = _make_client(handler, base_url="https://cp.example.com/")
        client.fetch_runtime_assignments()
        assert captured["url"] == "https://cp.example.com/api/v1/me/agent-runtime"


class TestResponseHandling:
    def test_200_parses_assignments_and_returns_etag(self):
        def handler(request):
            return _ok_response(assignments=[_assignment_payload(revision=3)], etag='"v3"')

        client = _make_client(handler)
        result = client.fetch_runtime_assignments()
        assert len(result.assignments) == 1
        assert result.assignments[0].revision == 3
        assert result.etag == '"v3"'
        assert result.not_modified is False

    def test_304_returns_not_modified_with_no_assignments(self):
        def handler(request):
            return httpx.Response(304, headers={"ETag": '"v1"'})

        client = _make_client(handler)
        result = client.fetch_runtime_assignments(etag='"v1"')
        assert result.not_modified is True
        assert result.assignments == ()
        assert result.status_code == 304
        assert result.etag == '"v1"'

    def test_200_with_empty_assignments_list(self):
        def handler(request):
            return _ok_response(assignments=[], etag='"empty"')

        client = _make_client(handler)
        result = client.fetch_runtime_assignments()
        assert result.assignments == ()
        assert result.etag == '"empty"'

    def test_malformed_response_missing_assignments_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True}, headers={"ETag": '"x"'})

        client = _make_client(handler)
        with pytest.raises(ControlPlaneError):
            client.fetch_runtime_assignments()


class TestRetry:
    def test_retries_on_5xx_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503)
            return _ok_response()

        client = _make_client(handler, max_retries=3)
        result = client.fetch_runtime_assignments()
        assert calls["n"] == 2
        assert result.status_code == 200

    def test_retries_on_timeout_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("simulated read timeout")
            return _ok_response()

        client = _make_client(handler, max_retries=3)
        result = client.fetch_runtime_assignments()
        assert calls["n"] == 2
        assert result.status_code == 200

    def test_gives_up_after_max_retries_on_5xx(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(503)

        client = _make_client(handler, max_retries=2)
        with pytest.raises(TransientFailureError):
            client.fetch_runtime_assignments()
        # 1 initial attempt + 2 retries
        assert calls["n"] == 3

    def test_gives_up_after_max_retries_on_timeout(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            raise httpx.ConnectTimeout("simulated")

        client = _make_client(handler, max_retries=1)
        with pytest.raises(TransientFailureError):
            client.fetch_runtime_assignments()
        assert calls["n"] == 2

    def test_uses_exponential_backoff(self):
        sleeps = []
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(500)

        client = _make_client(
            handler, max_retries=3, sleep=lambda s: sleeps.append(s),
            retry_base_delay=0.1, retry_backoff_factor=2.0,
        )
        with pytest.raises(TransientFailureError):
            client.fetch_runtime_assignments()
        # backoff: 0.1, 0.2, 0.4
        assert sleeps == [0.1, 0.2, 0.4]


class TestNoRetryOnAuthErrors:
    def test_401_raises_auth_error_without_retry(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(401)

        client = _make_client(handler, max_retries=3)
        with pytest.raises(AuthenticationError):
            client.fetch_runtime_assignments()
        assert calls["n"] == 1

    def test_403_raises_auth_error_without_retry(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(403)

        client = _make_client(handler, max_retries=3)
        with pytest.raises(AuthenticationError):
            client.fetch_runtime_assignments()
        assert calls["n"] == 1

    def test_other_4xx_raises_without_retry(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(404)

        client = _make_client(handler, max_retries=3)
        with pytest.raises(UnexpectedStatusError):
            client.fetch_runtime_assignments()
        assert calls["n"] == 1


class TestConstruction:
    def test_empty_token_rejected(self):
        with pytest.raises(ValueError):
            ControlPlaneClient(
                base_url="https://cp.example.com", token="", device_id="device-001",
                http_client=httpx.Client(transport=httpx.MockTransport(lambda r: _ok_response())),
            )

    def test_empty_device_id_rejected(self):
        with pytest.raises(ValueError):
            ControlPlaneClient(
                base_url="https://cp.example.com", token=TOKEN, device_id="",
                http_client=httpx.Client(transport=httpx.MockTransport(lambda r: _ok_response())),
            )


class TestRedaction:
    def test_logs_do_not_leak_token_or_manifest(self, caplog):
        def handler(request):
            return _ok_response()

        client = _make_client(handler)
        with caplog.at_level(logging.DEBUG, logger="hermes_managed.control_plane"):
            client.fetch_runtime_assignments()

        text = caplog.text
        assert TOKEN not in text
        assert MANIFEST_CANARY not in text
        assert "Bearer" not in text  # never log the auth header

    def test_logs_do_not_leak_manifest_on_retry_failure(self, caplog):
        def handler(request):
            return httpx.Response(503)

        client = _make_client(handler, max_retries=1)
        with caplog.at_level(logging.DEBUG, logger="hermes_managed.control_plane"):
            with pytest.raises(TransientFailureError):
                client.fetch_runtime_assignments()

        assert TOKEN not in caplog.text
        assert MANIFEST_CANARY not in caplog.text

    def test_logs_assignment_metadata_without_manifest(self, caplog):
        def handler(request):
            return _ok_response(assignments=[_assignment_payload(revision=7)])

        client = _make_client(handler)
        with caplog.at_level(logging.DEBUG, logger="hermes_managed.control_plane"):
            client.fetch_runtime_assignments()

        text = caplog.text
        # Non-sensitive metadata is logged for operability...
        assert "7" in text
        # ...but never the manifest canary.
        assert MANIFEST_CANARY not in text
