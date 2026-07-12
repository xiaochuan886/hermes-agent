"""Tests for hermes_managed.profile_applier — atomic managed-profile application.

Covers the frozen ownership and atomicity contract from the MVP plan (Task 6):

* the applier owns only ``<profile>/managed/`` and ``<profile>/skills/enterprise/``;
* it never deletes ``sessions/``, ``memories/``, ``skills/personal/`` or
  ``skills/learned/``;
* writes go to a sibling staging directory, are fsynced, then atomically
* published so a crash or failed replace leaves the previous valid state intact
  and never a half-readable config;
* revision semantics: stale rejected, idempotent re-apply is a no-op, same
  revision with different content fails closed, rollback (higher revision,
  older versionId) is allowed;
* revocation writes a tombstone / disabled state without deleting personal data;
* only Manifest-declared enterprise skills are materialized, path-traversal
  names are rejected, and symlinks cannot let writes escape the profile.
"""

import dataclasses
import json
import os
from pathlib import Path

import pytest

from hermes_managed.contracts import (
    ChecksumMismatchError,
    RuntimeAssignment,
    canonical_json_bytes,
    parse_runtime_assignment,
    sha256_hex,
)
from hermes_managed.profile_applier import (
    ApplyResult,
    PathTraversalError,
    ProfileApplier,
    ProfileStateError,
    ProfileTamperError,
    RevisionConflictError,
    StaleRevisionError,
)

FIXED_CLOCK = "2026-07-12T00:00:00Z"


def _policy() -> dict:
    policy = {
        "mode": "ENTERPRISE_MANAGED",
        "allowedModels": ["enterprise/deepseek-chat"],
        "defaultModel": "enterprise/deepseek-chat",
        "fallbackModels": [],
        "localProviderAllowed": False,
        "policyVersion": "v1",
    }
    policy["policySha256"] = sha256_hex(canonical_json_bytes(policy))
    return policy


def _manifest(skills=None, version="1.0.0") -> dict:
    if skills is None:
        skills = [
            {
                "slug": "summarize",
                "files": [{"path": "SKILL.md", "content": "# Summarize\n"}],
            }
        ]
    return {
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "version": version,
        "modelPolicy": {
            "mode": "ENTERPRISE_MANAGED",
            "allowedModels": ["enterprise/deepseek-chat"],
            "defaultModel": "enterprise/deepseek-chat",
            "fallbackModels": [],
            "localProvider": {"allowed": False},
        },
        "workspace": {"roots": ["/workspace/reference"]},
        "tools": {"allowed": ["read_file"], "approvalRequired": ["read_file"]},
        "enterpriseSkills": skills,
    }


def runtime_assignment(
    *,
    revision=2,
    version_id=19,
    revoked=False,
    revoked_at=None,
    skills=None,
    version="1.0.0",
    assignment_id=20,
) -> RuntimeAssignment:
    manifest = _manifest(skills=skills, version=version)
    payload = {
        "assignmentId": assignment_id,
        "revision": revision,
        "templateId": 10,
        "versionId": version_id,
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "manifest": manifest,
        "manifestSha256": sha256_hex(canonical_json_bytes(manifest)),
        "effectiveModelPolicy": _policy(),
        "revoked": revoked,
        "revokedAt": revoked_at,
    }
    return parse_runtime_assignment(payload)


def _read_assignment_json(profile: Path) -> dict:
    return json.loads((profile / "managed" / "assignment.json").read_text())


def _seed_personal_content(profile: Path) -> dict:
    """Create personal/learned/session/memory content that must be preserved."""
    paths = {
        "personal": profile / "skills" / "personal" / "mine" / "SKILL.md",
        "learned": profile / "skills" / "learned" / "auto" / "SKILL.md",
        "session": profile / "sessions" / "s1.json",
        "memory": profile / "memories" / "m1.json",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(p.name)
    return paths


def _assert_personal_preserved(profile: Path, paths: dict) -> None:
    for p in paths.values():
        assert p.exists(), f"personal content lost: {p}"
        assert p.read_text() == p.name


# --------------------------------------------------------------------------- #


class TestInitialApply:
    def test_apply_writes_managed_layer_and_preserves_personal_content(self, tmp_path):
        profile = tmp_path / "profiles" / "reference-assistant"
        personal = profile / "skills" / "personal" / "mine" / "SKILL.md"
        personal.parent.mkdir(parents=True)
        personal.write_text("personal")

        result = ProfileApplier(profile).apply(
            runtime_assignment(revision=2), clock=lambda: FIXED_CLOCK
        )

        assert result.revision == 2
        assert result.status == "initial"
        assert (profile / "managed" / "assignment.json").exists()
        assert personal.read_text() == "personal"

    def test_managed_is_symlink_into_versions_dir(self, tmp_path):
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(runtime_assignment(revision=2))

        managed = profile / "managed"
        assert managed.is_symlink()
        resolved = managed.resolve()
        assert resolved.is_relative_to((profile / ".managed-versions").resolve())
        assert (resolved / "assignment.json").exists()
        assert (resolved / "manifest.json").exists()
        assert (resolved / "policy.json").exists()

    def test_assignment_json_records_metadata(self, tmp_path):
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(
            runtime_assignment(revision=2, version_id=19), clock=lambda: FIXED_CLOCK
        )
        record = _read_assignment_json(profile)
        assert record["revision"] == 2
        assert record["version_id"] == 19
        assert record["assignment_id"] == 20
        assert record["revoked"] is False
        assert record["applied_at"] == FIXED_CLOCK
        assert "content_key" in record

    def test_apply_result_is_frozen(self, tmp_path):
        profile = tmp_path / "profile"
        result = ProfileApplier(profile).apply(runtime_assignment(revision=2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.revision = 99  # type: ignore[misc]


class TestRevisionSemantics:
    def test_update_to_higher_revision(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        result = applier.apply(runtime_assignment(revision=3, version_id=20))

        assert result.revision == 3
        assert result.status == "updated"
        assert _read_assignment_json(profile)["revision"] == 3

    def test_idempotent_reapply_is_noop(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        first = applier.apply(runtime_assignment(revision=2, version_id=19))
        second = applier.apply(runtime_assignment(revision=2, version_id=19))

        assert second.status == "idempotent"
        assert second.content_key == first.content_key
        # The version directory was not rebuilt.
        assert _read_assignment_json(profile)["applied_at"] == first.applied_at

    def test_stale_revision_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=3, version_id=20))
        with pytest.raises(StaleRevisionError):
            applier.apply(runtime_assignment(revision=2, version_id=19))
        assert _read_assignment_json(profile)["revision"] == 3

    def test_same_revision_different_content_fails_closed(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        with pytest.raises(RevisionConflictError):
            applier.apply(runtime_assignment(revision=2, version_id=20))
        assert _read_assignment_json(profile)["version_id"] == 19

    def test_rollback_to_older_version_id_with_higher_revision(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        # v2 (version_id=20) applied at revision 1
        applier.apply(runtime_assignment(revision=1, version_id=20))
        # rollback to v1 (version_id=19) at higher revision 2
        result = applier.apply(runtime_assignment(revision=2, version_id=19))

        assert result.revision == 2
        assert result.version_id == 19
        assert result.status == "updated"
        record = _read_assignment_json(profile)
        assert record["revision"] == 2
        assert record["version_id"] == 19


class TestAtomicity:
    def test_simulated_write_failure_preserves_old_state(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        def boom(*a, **k):
            raise OSError("simulated write failure")

        monkeypatch.setattr(applier, "_materialize", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))

        # Old managed state is intact and readable.
        assert _read_assignment_json(profile)["revision"] == 2

    def test_simulated_replace_failure_preserves_old_state(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        def boom(*a, **k):
            raise OSError("simulated replace failure")

        import hermes_managed.profile_applier as pa_mod

        monkeypatch.setattr(pa_mod, "_atomic_replace", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))

        # managed symlink was never swapped — old revision still active.
        assert _read_assignment_json(profile)["revision"] == 2
        # No orphaned "managed" entry left pointing at the new content.
        managed = profile / "managed"
        assert managed.is_symlink()
        assert json.loads((managed.resolve() / "assignment.json").read_text())["revision"] == 2

    def test_no_staging_or_swap_leftovers_after_success(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        applier.apply(runtime_assignment(revision=3, version_id=20))

        versions = list((profile / ".managed-versions").iterdir())
        # Only the active version dir remains; staging/swap temp names are gone.
        assert all(not n.name.startswith(".staging") for n in versions)
        assert all(not n.name.startswith(".managed.swap") for n in versions)
        assert all(not n.name.startswith(".") for n in versions)
        assert len(versions) == 1


class TestRevocation:
    def test_revoke_writes_tombstone_and_disables_profile(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=1, version_id=19))

        result = applier.apply(
            runtime_assignment(revision=2, version_id=19, revoked=True, revoked_at="2026-07-12T10:00:00Z")
        )

        assert result.revoked is True
        assert result.status == "revoked"
        assert (profile / "managed" / "tombstone.json").exists()
        record = _read_assignment_json(profile)
        assert record["revoked"] is True
        assert record["revoked_at"] == "2026-07-12T10:00:00Z"
        assert ProfileApplier.is_enabled(profile) is False
        _assert_personal_preserved(profile, paths)

    def test_normal_apply_leaves_profile_enabled(self, tmp_path):
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(runtime_assignment(revision=2, version_id=19))
        assert ProfileApplier.is_enabled(profile) is True

    def test_is_enabled_false_when_no_managed_state(self, tmp_path):
        profile = tmp_path / "profile"
        assert ProfileApplier.is_enabled(profile) is False

    def test_revoke_does_not_delete_enterprise_skills_directory(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=1, version_id=19))
        applier.apply(
            runtime_assignment(revision=2, version_id=19, revoked=True, revoked_at="2026-07-12T10:00:00Z")
        )
        # skills/enterprise still exists (as the tracking symlink) but is disabled.
        assert (profile / "skills" / "enterprise").is_symlink()


class TestOwnershipBoundary:
    def test_personal_learned_session_memory_all_preserved(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        ProfileApplier(profile).apply(runtime_assignment(revision=2, version_id=19))
        _assert_personal_preserved(profile, paths)

    def test_personal_content_preserved_across_update(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_personal_preserved(profile, paths)

    def test_managed_and_enterprise_are_the_only_managed_paths(self, tmp_path):
        profile = tmp_path / "profile"
        _seed_personal_content(profile)
        ProfileApplier(profile).apply(runtime_assignment(revision=2, version_id=19))

        # The applier created exactly these managed entries.
        assert (profile / "managed").is_symlink()
        assert (profile / "skills" / "enterprise").is_symlink()
        assert (profile / ".managed-versions").is_dir()
        # Personal directories are untouched real directories.
        assert (profile / "skills" / "personal").is_dir()
        assert not (profile / "skills" / "personal").is_symlink()
        assert (profile / "skills" / "learned").is_dir()
        assert not (profile / "skills" / "learned").is_symlink()


class TestEnterpriseSkills:
    def test_only_manifest_declared_skills_written(self, tmp_path):
        skills = [
            {"slug": "alpha", "files": [{"path": "SKILL.md", "content": "alpha"}]},
            {"slug": "beta", "files": [{"path": "SKILL.md", "content": "beta"}]},
        ]
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))

        ent = profile / "skills" / "enterprise"
        assert (ent / "alpha" / "SKILL.md").read_text() == "alpha"
        assert (ent / "beta" / "SKILL.md").read_text() == "beta"
        assert sorted(p.name for p in ent.iterdir()) == ["alpha", "beta"]

    def test_enterprise_skill_removal_only_affects_enterprise(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(
            runtime_assignment(
                revision=1,
                skills=[
                    {"slug": "alpha", "files": [{"path": "SKILL.md", "content": "alpha-v1"}]},
                    {"slug": "beta", "files": [{"path": "SKILL.md", "content": "beta-v1"}]},
                ],
            )
        )
        # Update: drop "beta", keep "alpha" with new content.
        applier.apply(
            runtime_assignment(
                revision=2,
                skills=[
                    {"slug": "alpha", "files": [{"path": "SKILL.md", "content": "alpha-v2"}]},
                ],
            )
        )

        ent = profile / "skills" / "enterprise"
        assert (ent / "alpha" / "SKILL.md").read_text() == "alpha-v2"
        assert not (ent / "beta").exists()
        _assert_personal_preserved(profile, paths)

    def test_enterprise_skill_tracks_managed_swap(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(
            runtime_assignment(
                revision=1,
                skills=[{"slug": "alpha", "files": [{"path": "SKILL.md", "content": "v1"}]}],
            )
        )
        assert (profile / "skills" / "enterprise" / "alpha" / "SKILL.md").read_text() == "v1"
        applier.apply(
            runtime_assignment(
                revision=2,
                skills=[{"slug": "alpha", "files": [{"path": "SKILL.md", "content": "v2"}]}],
            )
        )
        assert (profile / "skills" / "enterprise" / "alpha" / "SKILL.md").read_text() == "v2"

    def test_nested_skill_file_paths(self, tmp_path):
        skills = [
            {
                "slug": "alpha",
                "files": [
                    {"path": "SKILL.md", "content": "head"},
                    {"path": "scripts/run.py", "content": "print(1)"},
                ],
            }
        ]
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))
        ent = profile / "skills" / "enterprise"
        assert (ent / "alpha" / "SKILL.md").read_text() == "head"
        assert (ent / "alpha" / "scripts" / "run.py").read_text() == "print(1)"


class TestSecurity:
    def test_path_traversal_slug_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [{"slug": "../evil", "files": [{"path": "SKILL.md", "content": "x"}]}]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))
        assert not (profile / "managed").exists()

    def test_path_traversal_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [
            {"slug": "alpha", "files": [{"path": "../../etc/passwd", "content": "x"}]}
        ]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))
        assert not (profile / "managed").exists()

    def test_absolute_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [
            {"slug": "alpha", "files": [{"path": "/etc/passwd", "content": "x"}]}
        ]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))

    def test_backslash_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [
            {"slug": "alpha", "files": [{"path": "..\\evil", "content": "x"}]}
        ]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))

    def test_checksum_mismatch_rejected_at_apply(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        # Construct an assignment whose manifest_sha256 does not match (bypasses
        # parser validation via dataclasses.replace).
        good = runtime_assignment(revision=3, version_id=20)
        tampered = dataclasses.replace(good, manifest_sha256="0" * 64)
        with pytest.raises(ChecksumMismatchError):
            applier.apply(tampered)
        # Old state preserved.
        assert _read_assignment_json(profile)["revision"] == 2

    def test_symlink_managed_escaping_profile_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        profile.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (profile / "managed").parent.mkdir(parents=True, exist_ok=True)
        os.symlink(outside, profile / "managed")
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))
        # The outside directory was not written to.
        assert list(outside.iterdir()) == []

    def test_symlink_enterprise_escaping_profile_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        (profile / "skills").mkdir(parents=True)
        outside = tmp_path / "outside-ent"
        outside.mkdir()
        os.symlink(outside, profile / "skills" / "enterprise")
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))
        assert list(outside.iterdir()) == []

    def test_real_directory_managed_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        (profile / "managed").mkdir(parents=True)
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))

    def test_managed_versions_symlink_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        outside = tmp_path / "outside-versions"
        outside.mkdir()
        (profile).mkdir(parents=True, exist_ok=True)
        os.symlink(outside, profile / ".managed-versions")
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))
        assert list(outside.iterdir()) == []


class TestManagedStateReader:
    def test_read_state_returns_current_metadata(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        state = ProfileApplier.read_state(profile)
        assert state is not None
        assert state.revision == 2
        assert state.version_id == 19
        assert state.revoked is False

    def test_read_state_none_when_unmanaged(self, tmp_path):
        profile = tmp_path / "profile"
        assert ProfileApplier.read_state(profile) is None
