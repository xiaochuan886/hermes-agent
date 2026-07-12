"""Tests for hermes_managed.contracts — strict RuntimeAssignment parsing.

Covers the frozen contract decisions from the MVP plan (Task 6):

* required fields must be present and correctly typed;
* ``manifest_sha256`` is verified against the project canonical-JSON digest;
* ``effectiveModelPolicy.policySha256`` is verified the same way;
* checksum mismatch is never silently accepted;
* error messages must be explicit but must not leak manifest content or tokens.
"""

import dataclasses

import pytest

from hermes_managed.contracts import (
    ChecksumMismatchError,
    FieldTypeError,
    ManagedContractError,
    MissingRequiredFieldError,
    RuntimeAssignment,
    canonical_json_bytes,
    parse_runtime_assignment,
    sha256_hex,
)

# A canary planted inside the manifest / policy. No error message may contain it.
MANIFEST_CANARY = "MANIFEST_SECRET_CANARY_4f3a"
POLICY_CANARY = "POLICY_SECRET_CANARY_9c1b"


def _base_manifest() -> dict:
    return {
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "version": "1.0.0",
        "modelPolicy": {
            "mode": "ENTERPRISE_MANAGED",
            "allowedModels": ["enterprise/deepseek-chat"],
            "defaultModel": "enterprise/deepseek-chat",
            "fallbackModels": [],
            "localProvider": {"allowed": False},
        },
        "workspace": {"roots": ["/workspace/reference"]},
        "tools": {"allowed": ["read_file"], "approvalRequired": ["read_file"]},
        "enterpriseSkills": [
            {
                "slug": "summarize",
                "files": [{"path": "SKILL.md", "content": MANIFEST_CANARY}],
            }
        ],
        "note": MANIFEST_CANARY,
    }


def _base_policy() -> dict:
    policy = {
        "mode": "ENTERPRISE_MANAGED",
        "allowedModels": ["enterprise/deepseek-chat"],
        "defaultModel": "enterprise/deepseek-chat",
        "fallbackModels": ["enterprise/gpt-5"],
        "localProviderAllowed": False,
        "policyVersion": "v1",
        "note": POLICY_CANARY,
    }
    policy["policySha256"] = sha256_hex(canonical_json_bytes(policy))
    return policy


def _base_payload(**overrides) -> dict:
    manifest = _base_manifest()
    policy = _base_policy()
    payload = {
        "assignmentId": 20,
        "revision": 2,
        "templateId": 10,
        "versionId": 19,
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "manifest": manifest,
        "manifestSha256": sha256_hex(canonical_json_bytes(manifest)),
        "effectiveModelPolicy": policy,
        "revoked": False,
        "revokedAt": None,
    }
    payload.update(overrides)
    return payload


class TestCanonicalJson:
    def test_keys_sorted_recursively_and_compact(self):
        a = {"b": {"d": 1, "c": 2}, "a": 3}
        b = {"a": 3, "b": {"c": 2, "d": 1}}
        assert canonical_json_bytes(a) == canonical_json_bytes(b)

    def test_compact_separators(self):
        assert canonical_json_bytes({"a": 1, "b": 2}) == b'{"a":1,"b":2}'

    def test_unicode_preserved_as_utf8(self):
        assert canonical_json_bytes({"k": "中文"}) == '{"k":"中文"}'.encode("utf-8")

    def test_sha256_hex_is_64_lowercase_hex(self):
        h = sha256_hex(b"")
        assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert len(h) == 64


class TestParseValidAssignment:
    def test_populates_all_fields(self):
        assignment = parse_runtime_assignment(_base_payload())

        assert assignment.assignment_id == 20
        assert assignment.revision == 2
        assert assignment.template_id == 10
        assert assignment.version_id == 19
        assert assignment.template_slug == "reference-assistant"
        assert assignment.display_name == "Reference Assistant"
        assert assignment.manifest["templateSlug"] == "reference-assistant"
        assert assignment.manifest_sha256 == sha256_hex(
            canonical_json_bytes(_base_manifest())
        )
        assert assignment.effective_model_policy["policyVersion"] == "v1"
        assert assignment.revoked is False
        assert assignment.revoked_at is None

    def test_revoked_assignment_with_timestamp(self):
        payload = _base_payload(revoked=True, revokedAt="2026-07-12T10:00:00Z")
        assignment = parse_runtime_assignment(payload)
        assert assignment.revoked is True
        assert assignment.revoked_at == "2026-07-12T10:00:00Z"

    def test_revoked_at_defaults_to_none_when_absent(self):
        payload = _base_payload()
        payload.pop("revokedAt")
        assignment = parse_runtime_assignment(payload)
        assert assignment.revoked_at is None

    def test_assignment_is_frozen(self):
        assignment = parse_runtime_assignment(_base_payload())
        with pytest.raises(dataclasses.FrozenInstanceError):
            assignment.revision = 99  # type: ignore[misc]

    def test_unknown_top_level_fields_are_ignored(self):
        payload = _base_payload(extraServerField="ignore-me")
        assignment = parse_runtime_assignment(payload)
        assert assignment.revision == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("assignmentId", None),
        ("revision", None),
        ("templateId", None),
        ("versionId", None),
        ("templateSlug", None),
        ("displayName", None),
        ("manifest", None),
        ("manifestSha256", None),
        ("effectiveModelPolicy", None),
        ("revoked", None),
    ],
)
class TestMissingRequiredField:
    def test_missing_or_null_raises_without_leaking(self, field, value):
        payload = _base_payload()
        if value is None and field not in payload:
            return
        if value is None:
            del payload[field]
        else:
            payload[field] = value

        with pytest.raises((MissingRequiredFieldError, FieldTypeError)) as exc_info:
            parse_runtime_assignment(payload)

        message = str(exc_info.value)
        assert MANIFEST_CANARY not in message
        assert POLICY_CANARY not in message
        # Field name surfaces in the message so the caller can act, but no data.
        assert field.lower() in message.lower()


@pytest.mark.parametrize(
    "field,bad_value,expected",
    [
        ("assignmentId", "20", FieldTypeError),
        ("revision", "2", FieldTypeError),
        ("revision", 2.5, FieldTypeError),
        ("templateId", "10", FieldTypeError),
        ("versionId", "19", FieldTypeError),
        ("templateSlug", 123, FieldTypeError),
        ("displayName", 123, FieldTypeError),
        ("manifest", ["not", "a", "dict"], FieldTypeError),
        ("manifestSha256", 12345, FieldTypeError),
        ("effectiveModelPolicy", "not-a-dict", FieldTypeError),
        ("revoked", "true", FieldTypeError),
        ("revoked", 1, FieldTypeError),
        ("revokedAt", 12345, FieldTypeError),
    ],
)
def test_wrong_field_type_rejected(field, bad_value, expected):
    payload = _base_payload(**{field: bad_value})
    with pytest.raises(expected) as exc_info:
        parse_runtime_assignment(payload)
    assert MANIFEST_CANARY not in str(exc_info.value)


@pytest.mark.parametrize("field", ["assignmentId", "revision", "templateId", "versionId"])
def test_bool_rejected_for_integer_fields(field):
    # bool is a subclass of int in Python; a JSON true/false is not a valid id.
    payload = _base_payload(**{field: True})
    with pytest.raises(FieldTypeError):
        parse_runtime_assignment(payload)


@pytest.mark.parametrize(
    "bad_sha",
    [
        "abc",
        "ZZZZ" * 16,  # wrong length and non-hex
        "X" * 64,  # non-hex, right length
        "A" * 64,  # uppercase rejected (project convention is lowercase)
    ],
)
def test_invalid_manifest_sha_format_rejected(bad_sha):
    payload = _base_payload(manifestSha256=bad_sha)
    with pytest.raises((FieldTypeError, ChecksumMismatchError)) as exc_info:
        parse_runtime_assignment(payload)
    assert MANIFEST_CANARY not in str(exc_info.value)


class TestChecksumVerification:
    def test_manifest_checksum_mismatch_raises_without_leaking(self):
        manifest = _base_manifest()
        # Correct digest is overwritten with a valid-looking but wrong digest.
        payload = _base_payload(manifestSha256="0" * 64, manifest=manifest)
        with pytest.raises(ChecksumMismatchError) as exc_info:
            parse_runtime_assignment(payload)
        message = str(exc_info.value)
        assert MANIFEST_CANARY not in message
        # The real digest must never appear in the error either.
        assert sha256_hex(canonical_json_bytes(manifest)) not in message

    def test_policy_checksum_mismatch_raises_without_leaking(self):
        policy = _base_policy()
        policy["policySha256"] = "1" * 64  # valid format, wrong value
        payload = _base_payload(effectiveModelPolicy=policy)
        with pytest.raises(ChecksumMismatchError) as exc_info:
            parse_runtime_assignment(payload)
        assert POLICY_CANARY not in str(exc_info.value)

    def test_missing_policy_sha256_rejected(self):
        policy = _base_policy()
        del policy["policySha256"]
        payload = _base_payload(effectiveModelPolicy=policy)
        with pytest.raises(MissingRequiredFieldError) as exc_info:
            parse_runtime_assignment(payload)
        assert POLICY_CANARY not in str(exc_info.value)

    def test_policy_sha_bad_format_rejected(self):
        policy = _base_policy()
        policy["policySha256"] = "not-hex"
        payload = _base_payload(effectiveModelPolicy=policy)
        with pytest.raises((FieldTypeError, ChecksumMismatchError)) as exc_info:
            parse_runtime_assignment(payload)
        assert POLICY_CANARY not in str(exc_info.value)

    def test_manifest_tamper_after_correct_sha_detected(self):
        # If the server signed one manifest but sent another, the digest must
        # not match — proving the checksum is over the received bytes.
        manifest = _base_manifest()
        good_sha = sha256_hex(canonical_json_bytes(manifest))
        tampered = {**manifest, "note": "tampered"}
        payload = _base_payload(manifest=tampered, manifestSha256=good_sha)
        with pytest.raises(ChecksumMismatchError):
            parse_runtime_assignment(payload)

    def test_all_errors_are_managed_contract_errors(self):
        for exc in (
            MissingRequiredFieldError("x"),
            FieldTypeError("x"),
            ChecksumMismatchError("x"),
        ):
            assert isinstance(exc, ManagedContractError)
