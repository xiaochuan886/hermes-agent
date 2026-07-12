"""Strict contracts for the Hermes managed runtime.

Defines :class:`RuntimeAssignment` and the project's canonical-JSON / SHA-256
checksum algorithm used to verify control-plane payloads.

Canonical JSON algorithm (project convention — the SkillHub control plane MUST
produce identical bytes when computing ``manifestSha256`` and
``effectiveModelPolicy.policySha256``)::

    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = sha256(canonical.encode("utf-8")).hexdigest()   # 64 lowercase hex

* keys are sorted at every nesting level;
* separators are compact (no whitespace);
* non-ASCII characters are encoded as UTF-8 bytes (``ensure_ascii=False``), not
  ``\\u`` escapes, so the digest is stable across runtimes;
* the digest is lowercase hexadecimal.

``effectiveModelPolicy.policySha256`` is computed over the policy object with
the ``policySha256`` field removed (it is not self-referential).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "ManagedContractError",
    "MissingRequiredFieldError",
    "FieldTypeError",
    "ChecksumMismatchError",
    "RuntimeAssignment",
    "canonical_json_bytes",
    "sha256_hex",
    "parse_runtime_assignment",
    "verify_manifest_checksum",
    "verify_policy_checksum",
]

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# --- exceptions ---------------------------------------------------------------


class ManagedContractError(Exception):
    """Base class for all managed-runtime contract failures.

    Subclass messages describe *what* was wrong (which field, which check) but
    never embed payload content — manifests, policies or credentials are not
    interpolated into message text.
    """


class MissingRequiredFieldError(ManagedContractError):
    """A required field was absent or null."""


class FieldTypeError(ManagedContractError):
    """A field had the wrong type or an invalid format (e.g. bad hex)."""


class ChecksumMismatchError(ManagedContractError):
    """A manifest or policy checksum did not match the recomputed digest."""


# --- canonical JSON + SHA-256 -------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """Recursively convert mappings (incl. ``MappingProxyType``) to plain dicts.

    ``json.dumps`` cannot serialize ``MappingProxyType``; this normalizes the
    structure without changing the canonical byte output (keys are still sorted
    by :func:`canonical_json_bytes`).
    """
    if isinstance(obj, Mapping):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def canonical_json_bytes(obj: Any) -> bytes:
    """Return the canonical UTF-8 byte encoding of ``obj``.

    Keys are sorted recursively, separators are compact, non-ASCII is preserved.
    """
    return json.dumps(
        _to_serializable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _require_hex64(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise FieldTypeError(f"field '{field}' must be a 64-char hex string")
    if not _HEX64_RE.match(value):
        raise FieldTypeError(f"field '{field}' must be 64 lowercase hex chars")
    return value


def _content_digest(obj: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical form of a mapping (used for manifest/policy)."""
    return sha256_hex(canonical_json_bytes(dict(obj)))


# --- RuntimeAssignment --------------------------------------------------------

# camelCase JSON key -> (attribute, validator).  ``revokedAt`` is optional.
_REQUIRED_FIELDS: tuple[tuple[str, str], ...] = (
    ("assignmentId", "assignment_id"),
    ("revision", "revision"),
    ("templateId", "template_id"),
    ("versionId", "version_id"),
    ("templateSlug", "template_slug"),
    ("displayName", "display_name"),
    ("manifest", "manifest"),
    ("manifestSha256", "manifest_sha256"),
    ("effectiveModelPolicy", "effective_model_policy"),
    ("revoked", "revoked"),
)


@dataclass(frozen=True)
class RuntimeAssignment:
    """An immutable assignment of a template version to a device/profile.

    Attributes mirror the SkillHub ``RuntimeAssignmentResponse`` (Task 4) using
    snake_case.  ``manifest`` and ``effective_model_policy`` are kept as the
    parsed objects received from the control plane.
    """

    assignment_id: int
    revision: int
    template_id: int
    version_id: int
    template_slug: str
    display_name: str
    manifest: Mapping[str, Any]
    manifest_sha256: str
    effective_model_policy: Mapping[str, Any]
    revoked: bool
    revoked_at: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RuntimeAssignment":
        return parse_runtime_assignment(payload)


def _check_present(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise MissingRequiredFieldError(f"missing required field: {key}")
    value = payload[key]
    if value is None:
        raise MissingRequiredFieldError(f"required field is null: {key}")
    return value


def _check_int(value: Any, key: str) -> int:
    # bool is a subclass of int; a JSON true/false is not a valid integer id.
    if isinstance(value, bool) or not isinstance(value, int):
        raise FieldTypeError(f"field '{key}' must be an integer")
    return value


def _check_str(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise FieldTypeError(f"field '{key}' must be a string")
    return value


def parse_runtime_assignment(payload: Mapping[str, Any]) -> RuntimeAssignment:
    """Strictly parse a control-plane assignment payload.

    Raises :class:`MissingRequiredFieldError`, :class:`FieldTypeError` or
    :class:`ChecksumMismatchError` on any violation.  Error messages never
    contain manifest content, policy content or credentials.
    """
    if not isinstance(payload, Mapping):
        raise FieldTypeError("assignment payload must be a JSON object")

    assignment_id = _check_int(_check_present(payload, "assignmentId"), "assignmentId")
    revision = _check_int(_check_present(payload, "revision"), "revision")
    if revision < 1:
        raise FieldTypeError("field 'revision' must be >= 1")
    template_id = _check_int(_check_present(payload, "templateId"), "templateId")
    version_id = _check_int(_check_present(payload, "versionId"), "versionId")
    template_slug = _check_str(_check_present(payload, "templateSlug"), "templateSlug")
    display_name = _check_str(_check_present(payload, "displayName"), "displayName")

    manifest = _check_present(payload, "manifest")
    if not isinstance(manifest, Mapping):
        raise FieldTypeError("field 'manifest' must be a JSON object")
    manifest = dict(manifest)

    manifest_sha256 = _require_hex64(_check_present(payload, "manifestSha256"), "manifestSha256")

    policy = _check_present(payload, "effectiveModelPolicy")
    if not isinstance(policy, Mapping):
        raise FieldTypeError("field 'effectiveModelPolicy' must be a JSON object")
    policy = dict(policy)

    revoked_raw = _check_present(payload, "revoked")
    if not isinstance(revoked_raw, bool):
        raise FieldTypeError("field 'revoked' must be a boolean")
    revoked = revoked_raw

    revoked_at = payload.get("revokedAt")
    if revoked_at is not None and not isinstance(revoked_at, str):
        raise FieldTypeError("field 'revokedAt' must be a string or null")

    # Verify checksums against the received bytes — fail closed on mismatch.
    verify_manifest_checksum(manifest, manifest_sha256)
    verify_policy_checksum(policy)

    return RuntimeAssignment(
        assignment_id=assignment_id,
        revision=revision,
        template_id=template_id,
        version_id=version_id,
        template_slug=template_slug,
        display_name=display_name,
        manifest=MappingProxyType(manifest),
        manifest_sha256=manifest_sha256,
        effective_model_policy=MappingProxyType(policy),
        revoked=revoked,
        revoked_at=revoked_at,
    )


def verify_manifest_checksum(manifest: Mapping[str, Any], expected_sha256: str) -> None:
    """Raise :class:`ChecksumMismatchError` unless ``expected_sha256`` matches.

    Also rejects a non-64-lowercase-hex ``expected_sha256``.
    """
    _require_hex64(expected_sha256, "manifestSha256")
    actual = _content_digest(manifest)
    if not _const_time_eq(actual, expected_sha256):
        raise ChecksumMismatchError(
            "manifest_sha256 does not match the canonical manifest digest"
        )


def verify_policy_checksum(policy: Mapping[str, Any]) -> None:
    """Verify ``effectiveModelPolicy.policySha256`` over the policy minus itself."""
    if "policySha256" not in policy or policy["policySha256"] is None:
        raise MissingRequiredFieldError(
            "missing required field: effectiveModelPolicy.policySha256"
        )
    expected = _require_hex64(policy["policySha256"], "policySha256")
    content = {k: v for k, v in policy.items() if k != "policySha256"}
    actual = _content_digest(content)
    if not _const_time_eq(actual, expected):
        raise ChecksumMismatchError(
            "effectiveModelPolicy.policySha256 does not match the canonical policy digest"
        )


def _const_time_eq(a: str, b: str) -> bool:
    """Constant-time comparison of two hex strings (lengths already validated)."""
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))
