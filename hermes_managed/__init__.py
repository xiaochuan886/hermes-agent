"""Hermes managed runtime — applies immutable control-plane assignments to profiles.

This package owns the enterprise-managed layer of a Hermes profile:

* :mod:`hermes_managed.contracts` — strict parsing of ``RuntimeAssignment`` and
  canonical JSON / SHA-256 checksum verification.
* :mod:`hermes_managed.control_plane_client` — authenticated sync client for the
  SkillHub ``/api/v1/me/agent-runtime`` endpoint.
* :mod:`hermes_managed.profile_applier` — atomic, fail-closed application of an
  assignment to ``<profile>/managed/`` and ``<profile>/skills/enterprise/``.

The applier never touches ``sessions/``, ``memories/``, ``skills/personal/`` or
``skills/learned/``. See ``docs/superpowers/plans/2026-07-11-mvp-reference-agent-end-to-end.md``
(Task 6) for the frozen contract decisions.
"""

from hermes_managed.contracts import (
    ChecksumMismatchError,
    FieldTypeError,
    ManagedContractError,
    MissingRequiredFieldError,
    RuntimeAssignment,
    canonical_json_bytes,
    parse_runtime_assignment,
    sha256_hex,
    verify_manifest_checksum,
    verify_policy_checksum,
)

__all__ = [
    "ChecksumMismatchError",
    "FieldTypeError",
    "ManagedContractError",
    "MissingRequiredFieldError",
    "RuntimeAssignment",
    "canonical_json_bytes",
    "parse_runtime_assignment",
    "sha256_hex",
    "verify_manifest_checksum",
    "verify_policy_checksum",
]
