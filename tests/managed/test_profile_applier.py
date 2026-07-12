"""Tests for hermes_managed.profile_applier — atomic managed-profile application.

Covers the frozen ownership/atomicity contract (Task 6) and the P1 hardening:
cross-process locking, single commit point + rollback, TOCTOU/symlink defense
via pinned directory fds, and persistence ordering.
"""

import dataclasses
import errno
import json
import os
import subprocess
import sys
import time
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
    ProfileDurabilityUncertainError,
    ProfileLockTimeoutError,
    ProfileRollbackError,
    ProfileStateError,
    ProfileTamperError,
    RevisionConflictError,
    StaleRevisionError,
)

FIXED_CLOCK = "2026-07-12T00:00:00Z"
GOLDEN_POLICY_SHA = "5ea1e6720cc6dc3ef2cae5579c687dd98d80245bdf3a2ca89fb70e85c876899a"


def _policy() -> dict:
    # Golden vector 1 (framed checksum) — see test_contracts.GOLDEN_VECTORS.
    return {
        "mode": "ENTERPRISE_MANAGED",
        "allowedModels": ["enterprise/deepseek-chat"],
        "defaultModel": "enterprise/deepseek-chat",
        "fallbackModels": [],
        "localProviderAllowed": False,
        "policyVersion": "sha256:5ea1e6720cc6",
        "policySha256": GOLDEN_POLICY_SHA,
    }


def _manifest(skills=None, version="1.0.0") -> dict:
    if skills is None:
        skills = [
            {"slug": "summarize", "files": [{"path": "SKILL.md", "content": "# Summarize\n"}]}
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


def _active_version_key(profile: Path) -> str:
    return os.readlink(profile / "managed").split("/", 1)[1]


def _seed_personal_content(profile: Path) -> dict:
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


def _assert_old_state_intact(profile: Path, revision: int, paths: dict) -> None:
    """Pre-publish failure invariant: old assignment + skills + personal intact."""
    assert _read_assignment_json(profile)["revision"] == revision
    _assert_personal_preserved(profile, paths)
    assert ProfileApplier.read_state(profile).revision == revision


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
        assert record["policy_sha256"] == GOLDEN_POLICY_SHA
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
        applier.apply(runtime_assignment(revision=1, version_id=20))
        result = applier.apply(runtime_assignment(revision=2, version_id=19))
        assert result.revision == 2
        assert result.version_id == 19
        assert result.status == "updated"
        record = _read_assignment_json(profile)
        assert record["revision"] == 2
        assert record["version_id"] == 19


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
        assert (profile / "managed").is_symlink()
        assert (profile / "skills" / "enterprise").is_symlink()
        assert (profile / ".managed-versions").is_dir()
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
        assert (profile / "skills" / "enterprise").is_symlink()


class TestManagedStateReader:
    def test_read_state_returns_current_metadata(self, tmp_path):
        profile = tmp_path / "profile"
        ProfileApplier(profile).apply(runtime_assignment(revision=2, version_id=19))
        state = ProfileApplier.read_state(profile)
        assert state is not None
        assert state.revision == 2
        assert state.version_id == 19
        assert state.revoked is False

    def test_read_state_none_when_unmanaged(self, tmp_path):
        profile = tmp_path / "profile"
        assert ProfileApplier.read_state(profile) is None


class TestAtomicity:
    def test_simulated_write_failure_preserves_old_state(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        def boom(*a, **k):
            raise OSError("simulated write failure")

        monkeypatch.setattr(applier, "_materialize", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)

    def test_simulated_replace_failure_preserves_old_state(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        def boom(*a, **k):
            raise OSError("simulated managed replace failure")

        monkeypatch.setattr(applier, "_commit_managed_link", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)
        managed = profile / "managed"
        assert managed.is_symlink()
        assert json.loads((managed.resolve() / "assignment.json").read_text())["revision"] == 2

    def test_no_staging_or_swap_leftovers_after_success(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        applier.apply(runtime_assignment(revision=3, version_id=20))
        versions = list((profile / ".managed-versions").iterdir())
        names = [n.name for n in versions]
        # Only the active version dir + the apply lock remain.
        assert all(not n.startswith(".staging.") for n in names)
        assert all(not n.startswith(".managed.swap.") for n in names)
        assert all(not n.startswith(".enterprise.swap.") for n in names)
        non_lock = [n for n in names if n != ".apply.lock"]
        assert len(non_lock) == 1  # the active version dir
        # No swap temp files left in the profile root or skills dir either.
        assert not any(p.name.startswith(".managed.swap.") for p in profile.iterdir())
        assert not any(
            p.name.startswith(".enterprise.swap.") for p in (profile / "skills").iterdir()
        )


# --- fault injection (section 2) ---------------------------------------------


class TestFaultInjection:
    """Each pre-commit failure must leave the previous valid state intact and
    publish no new revision.  Post-commit cleanup failure must not fail apply."""

    def _setup(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(
            runtime_assignment(
                revision=2,
                version_id=19,
                skills=[{"slug": "old", "files": [{"path": "SKILL.md", "content": "old"}]}],
            )
        )
        return profile, paths, applier

    def test_enterprise_link_mkdir_failure(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)

        def boom(*a, **k):
            raise OSError("skills dir mkdir failure")

        monkeypatch.setattr(applier, "_open_skills_dir", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)
        assert (profile / "skills" / "enterprise" / "old" / "SKILL.md").read_text() == "old"

    def test_enterprise_symlink_create_failure(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)
        # Force the "first apply" branch by removing the existing enterprise link
        # so _prepare_enterprise_link tries to create it fresh.
        os.unlink(profile / "skills" / "enterprise")

        def boom(*a, **k):
            raise OSError("enterprise symlink create failure")

        monkeypatch.setattr(applier, "_create_enterprise_symlink", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)

    def test_enterprise_parent_fsync_failure(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)
        os.unlink(profile / "skills" / "enterprise")

        def boom(*a, **k):
            raise OSError("enterprise parent fsync failure")

        monkeypatch.setattr(applier, "_fsync_enterprise_parent", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)

    def test_managed_atomic_replace_failure(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)

        def boom(*a, **k):
            raise OSError("managed atomic replace failure")

        monkeypatch.setattr(applier, "_commit_managed_link", boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)
        # The new version dir was rolled back (no orphan).
        versions = [n.name for n in (profile / ".managed-versions").iterdir()]
        assert all("r00000000000000000003" not in n for n in versions)

    def test_post_publish_cleanup_failure_does_not_fail_apply(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)

        def boom(*a, **k):
            raise OSError("cleanup failure")

        monkeypatch.setattr(applier, "_cleanup_old_versions", boom)
        result = applier.apply(runtime_assignment(revision=3, version_id=20))
        assert result.revision == 3
        assert result.status == "updated"
        _assert_personal_preserved(profile, paths)

    def test_rollback_failure_is_fail_closed(self, tmp_path, monkeypatch):
        profile, paths, applier = self._setup(tmp_path)

        def publish_boom(*a, **k):
            raise OSError("commit failure")

        def rollback_boom(*a, **k):
            raise OSError("rollback failure")

        monkeypatch.setattr(applier, "_commit_managed_link", publish_boom)
        monkeypatch.setattr(applier, "_rollback", rollback_boom)
        with pytest.raises(ProfileRollbackError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        # managed was never swapped -> old state still active.
        assert _read_assignment_json(profile)["revision"] == 2
        _assert_personal_preserved(profile, paths)


# --- post-commit durability failure (commit/durability split) ----------------


class TestDurabilityFailure:
    """A post-commit fsync failure must NOT roll back the committed active state.

    The commit (managed symlink swap) and the durability sync (profile dir
    fsync) are separate: once the swap succeeds the new revision is active, so a
    subsequent fsync EIO is reported as ProfileDurabilityUncertainError without
    deleting the new version dir or restoring the old managed link.
    """

    def test_initial_apply_post_commit_fsync_eio(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)

        def boom(profile_fd):
            raise OSError(errno.EIO, "simulated profile dir fsync EIO")

        monkeypatch.setattr(applier, "_fsync_profile_dir", boom)
        with pytest.raises(ProfileDurabilityUncertainError):
            applier.apply(runtime_assignment(revision=2, version_id=19))
        # The new revision IS active: managed not dangling, version_dir present,
        # assignment.json readable at the new revision.
        managed = profile / "managed"
        assert managed.is_symlink()
        assert (managed.resolve() / "assignment.json").exists()
        assert _read_assignment_json(profile)["revision"] == 2
        # enterprise link still tracks managed.
        assert (profile / "skills" / "enterprise").is_symlink()
        _assert_personal_preserved(profile, paths)

    def test_update_post_commit_fsync_eio_preserves_new_and_idempotent_retry(
        self, tmp_path, monkeypatch
    ):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        rev2_key = _active_version_key(profile)

        def boom(profile_fd):
            raise OSError(errno.EIO, "simulated profile dir fsync EIO")

        monkeypatch.setattr(applier, "_fsync_profile_dir", boom)
        with pytest.raises(ProfileDurabilityUncertainError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        # managed points to rev3 (not dangling); rev3 version_dir preserved.
        managed = profile / "managed"
        assert managed.is_symlink()
        assert _read_assignment_json(profile)["revision"] == 3
        # rev2 NOT deleted: no cleanup runs in the durability-uncertain path.
        assert (profile / ".managed-versions" / rev2_key / "assignment.json").exists()
        # Retrying the same revision/content is safe and idempotent.
        result = ProfileApplier(profile).apply(runtime_assignment(revision=3, version_id=20))
        assert result.status == "idempotent"
        assert result.revision == 3

    def test_pre_commit_replace_failure_calls_rollback(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        rollback_calls = []

        def rollback_spy(*a, **k):
            rollback_calls.append(True)

        def commit_boom(*a, **k):
            raise OSError("commit replace failed")

        monkeypatch.setattr(applier, "_rollback", rollback_spy)
        monkeypatch.setattr(applier, "_commit_managed_link", commit_boom)
        with pytest.raises(OSError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        # Pre-commit failure DOES roll back and preserves the old active state.
        assert len(rollback_calls) == 1
        _assert_old_state_intact(profile, 2, paths)

    def test_post_commit_fsync_failure_never_calls_rollback(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))

        rollback_calls = []

        def rollback_spy(*a, **k):
            rollback_calls.append(True)

        def fsync_boom(profile_fd):
            raise OSError(errno.EIO, "simulated profile dir fsync EIO")

        monkeypatch.setattr(applier, "_rollback", rollback_spy)
        monkeypatch.setattr(applier, "_fsync_profile_dir", fsync_boom)
        with pytest.raises(ProfileDurabilityUncertainError):
            applier.apply(runtime_assignment(revision=3, version_id=20))
        # The committed new state must NOT be rolled back.
        assert rollback_calls == []
        assert _read_assignment_json(profile)["revision"] == 3


# --- cross-process concurrency (section 3) -----------------------------------
#
# Uses separate Python subprocesses (tests/managed/_apply_worker.py) so the
# fcntl apply lock is exercised between truly independent processes — stronger
# than threads and avoids fork()-from-a-multi-threaded-pytest-worker deadlocks.

_WORKER = Path(__file__).resolve().parent / "_apply_worker.py"
_ROOT = Path(__file__).resolve().parents[2]


def _run_worker(profile, mode, *args, timeout=30.0):
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, str(_WORKER), mode, str(profile), *map(str, args)],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(_ROOT)},
    )
    assert proc.returncode == 0, f"worker failed: {proc.stderr}"
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _run_pair(profile, args1, args2):
    import subprocess
    import sys as _sys

    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    cmd = lambda a: [_sys.executable, str(_WORKER), "apply", str(profile), *map(str, a)]
    p1 = subprocess.Popen(cmd(args1), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    p2 = subprocess.Popen(cmd(args2), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out1, _ = p1.communicate(timeout=30)
    out2, _ = p2.communicate(timeout=30)
    return [json.loads(out1.strip().splitlines()[-1]), json.loads(out2.strip().splitlines()[-1])]


class TestConcurrency:
    SKILLS = [{"slug": "alpha", "files": [{"path": "SKILL.md", "content": "a"}]}]

    def test_rev2_and_rev3_concurrent_final_is_rev3(self, tmp_path):
        profile = tmp_path / "profile"
        results = _run_pair(
            profile,
            (2, 19, json.dumps(self.SKILLS), 0),
            (3, 20, json.dumps(self.SKILLS), 0),
        )
        assert ProfileApplier.read_state(profile).revision == 3
        assert len(results) == 2  # both completed

    def test_rev3_in_progress_blocks_rev2_from_overwriting(self, tmp_path):
        profile = tmp_path / "profile"
        # Stagger: rev3 starts first and reaches its pre-commit pause (holding
        # the lock); rev2 starts afterwards and must block until rev3 commits.
        env = {**os.environ, "PYTHONPATH": str(_ROOT)}
        c1 = [sys.executable, str(_WORKER), "apply", str(profile),
              "3", "20", json.dumps(self.SKILLS), "0.5"]
        c2 = [sys.executable, str(_WORKER), "apply", str(profile),
              "2", "19", json.dumps(self.SKILLS), "0"]
        p1 = subprocess.Popen(c1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        time.sleep(0.25)  # rev3 now holds the lock mid-apply
        p2 = subprocess.Popen(c2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        o1, _ = p1.communicate(timeout=30)
        o2, _ = p2.communicate(timeout=30)
        r1 = json.loads(o1.strip().splitlines()[-1])
        r2 = json.loads(o2.strip().splitlines()[-1])
        assert r1.get("ok") is True, r1
        # rev2 must not overwrite rev3; it sees the committed rev3 and fails stale.
        assert r2.get("ok") is False
        assert "StaleRevisionError" in r2.get("error", "")
        assert ProfileApplier.read_state(profile).revision == 3

    def test_two_same_revision_same_content_concurrent_idempotent(self, tmp_path):
        profile = tmp_path / "profile"
        results = _run_pair(
            profile,
            (2, 19, json.dumps(self.SKILLS), 0),
            (2, 19, json.dumps(self.SKILLS), 0),
        )
        assert ProfileApplier.read_state(profile).revision == 2
        oks = [r for r in results if r.get("ok")]
        assert len(oks) == 2  # one initial, one idempotent

    def test_two_same_revision_different_content_one_conflicts(self, tmp_path):
        profile = tmp_path / "profile"
        skills_a = [{"slug": "a", "files": [{"path": "SKILL.md", "content": "a"}]}]
        skills_b = [{"slug": "b", "files": [{"path": "SKILL.md", "content": "b"}]}]
        results = _run_pair(
            profile,
            (2, 19, json.dumps(skills_a), 0),
            (2, 20, json.dumps(skills_b), 0),
        )
        oks = [r for r in results if r.get("ok")]
        errs = [r for r in results if not r.get("ok")]
        assert len(oks) == 1
        assert len(errs) == 1
        assert any("Conflict" in r.get("error", "") for r in errs)

    def test_no_staging_or_bad_active_link_left(self, tmp_path):
        profile = tmp_path / "profile"
        _run_pair(
            profile,
            (2, 19, json.dumps(self.SKILLS), 0),
            (3, 20, json.dumps(self.SKILLS), 0),
        )
        versions = list((profile / ".managed-versions").iterdir())
        assert all(not n.name.startswith(".staging.") for n in versions)
        assert all(not n.name.startswith(".managed.swap.") for n in versions)
        managed = profile / "managed"
        assert managed.is_symlink()
        assert (managed.resolve() / "assignment.json").exists()


# --- TOCTOU / symlink defense (section 4) ------------------------------------


class TestTOCTOU:
    def test_managed_versions_replaced_with_symlink_after_open(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        old_key = _active_version_key(profile)
        outside = tmp_path / "outside-mv"
        outside.mkdir()
        canary = outside / "CANARY"
        canary.write_text("secret")
        real = profile / ".managed-versions"

        def hook():
            # Replace the .managed-versions entry with a symlink to outside.
            os.rename(real, profile / ".managed-versions.real")
            os.symlink(outside, real)

        applier2 = ProfileApplier(profile)
        applier2._after_open_hook = hook
        with pytest.raises(ProfileTamperError):
            applier2.apply(runtime_assignment(revision=3, version_id=20))
        # The pinned versions_fd wrote staging to the original (renamed) dir,
        # never to the outside target.
        assert canary.read_text() == "secret"
        # Old assignment.json preserved in the renamed real versions dir.
        assert json.loads(
            (profile / ".managed-versions.real" / old_key / "assignment.json").read_text()
        )["revision"] == 2
        # skills/sessions/memories were not touched by this hook.
        _assert_personal_preserved(profile, paths)

    def test_skills_replaced_with_external_symlink_after_open(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        outside = tmp_path / "outside-skills"
        outside.mkdir()
        canary = outside / "CANARY"
        canary.write_text("secret")
        skills_path = profile / "skills"

        def hook():
            os.rename(skills_path, profile / "skills.real")
            os.symlink(outside, skills_path)

        applier2 = ProfileApplier(profile)
        applier2._after_open_hook = hook
        with pytest.raises(ProfileTamperError):
            applier2.apply(runtime_assignment(revision=3, version_id=20))
        assert canary.read_text() == "secret"
        # managed / .managed-versions untouched -> old state still readable.
        assert _read_assignment_json(profile)["revision"] == 2
        # personal content preserved in the renamed skills dir.
        assert (profile / "skills.real" / "personal" / "mine" / "SKILL.md").read_text() == "SKILL.md"
        assert (profile / "skills.real" / "learned" / "auto" / "SKILL.md").exists()

    def test_cleanup_does_not_follow_symlink_to_external(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(
            runtime_assignment(
                revision=1,
                version_id=19,
                skills=[{"slug": "a", "files": [{"path": "SKILL.md", "content": "a"}]}],
            )
        )
        # Plant a symlink inside .managed-versions pointing at an outside dir
        # that contains a canary file.
        outside = tmp_path / "outside-cleanup"
        outside.mkdir()
        canary = outside / "CANARY"
        canary.write_text("secret")
        os.symlink(outside, profile / ".managed-versions" / "evil-link")

        # Apply an update -> cleanup runs and must NOT follow the symlink.
        applier.apply(
            runtime_assignment(
                revision=2,
                version_id=20,
                skills=[{"slug": "a", "files": [{"path": "SKILL.md", "content": "a2"}]}],
            )
        )
        assert canary.read_text() == "secret"
        assert (profile / ".managed-versions" / "evil-link").is_symlink()
        assert ProfileApplier.read_state(profile).revision == 2

    def test_managed_link_replaced_before_commit_no_external_write(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        outside = tmp_path / "outside-managed"
        outside.mkdir()
        canary = outside / "CANARY"
        canary.write_text("secret")

        managed_path = profile / "managed"

        def hook():
            # Swap managed to an external symlink right before the commit.
            if managed_path.is_symlink():
                os.unlink(managed_path)
            os.symlink(outside, managed_path)

        applier2 = ProfileApplier(profile)
        applier2._pre_commit_hook = hook
        # The commit atomically reclaims the tampered managed symlink; the
        # external target is never written to.
        result = applier2.apply(runtime_assignment(revision=3, version_id=20))
        assert result.revision == 3
        assert canary.read_text() == "secret"
        assert (profile / "managed").is_symlink()
        assert json.loads((profile / "managed" / "assignment.json").read_text())["revision"] == 3
        _assert_personal_preserved(profile, paths)

    def test_enterprise_link_replaced_before_prepare_is_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        paths = _seed_personal_content(profile)
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        outside = tmp_path / "outside-ent2"
        outside.mkdir()
        canary = outside / "CANARY"
        canary.write_text("secret")

        ent_path = profile / "skills" / "enterprise"

        def hook():
            if ent_path.is_symlink():
                os.unlink(ent_path)
            os.symlink(outside, ent_path)

        applier2 = ProfileApplier(profile)
        applier2._after_open_hook = hook
        with pytest.raises(ProfileTamperError):
            applier2.apply(runtime_assignment(revision=3, version_id=20))
        _assert_old_state_intact(profile, 2, paths)
        assert canary.read_text() == "secret"

    def test_external_canaries_unchanged_across_all_vectors(self, tmp_path):
        # A normal apply must never touch outside canaries reachable via an
        # unrelated symlink planted inside the profile.
        profile = tmp_path / "profile"
        profile.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        c1 = outside / "c1"
        c1.write_text("one")
        os.symlink(outside, profile / ".canary-link")
        ProfileApplier(profile).apply(runtime_assignment(revision=2, version_id=19))
        assert c1.read_text() == "one"
        assert (profile / ".canary-link").is_symlink()


# --- persistence (section 5) -------------------------------------------------


class TestPersistence:
    def test_dir_fsync_tolerates_macos_einval(self, tmp_path):
        # macOS returns EINVAL for fsync on a directory fd; this must not raise.
        import hermes_managed.profile_applier as pa

        fd = os.open(str(tmp_path), os.O_RDONLY)
        try:
            pa._fsync_dir_fd(fd)  # must not raise
        finally:
            os.close(fd)

    def test_enterprise_parent_fsync_is_invoked(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        called = []

        def spy(skills_fd):
            called.append(True)
            return pa._fsync_dir_fd(skills_fd)

        import hermes_managed.profile_applier as pa

        monkeypatch.setattr(applier, "_fsync_enterprise_parent", spy)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        assert called  # fsync seam exercised


# --- FD ownership / leak regression ------------------------------------------


class TestFdLeak:
    """Every directory fd opened by the applier must be closed by the applier.

    Wraps the fd-returning helpers (and os.close) to record opens vs closes and
    asserts no fd is leaked across many revisions with many enterprise skills.
    """

    @staticmethod
    def _install_trackers(monkeypatch):
        import hermes_managed.profile_applier as pa

        opened, closed = set(), set()
        real_open_dir = pa._open_dir_fd
        real_subdir = pa._open_or_create_subdir
        real_lock = pa._open_lock_file
        real_close = os.close

        def w_open_dir(*a, **k):
            fd = real_open_dir(*a, **k)
            opened.add(fd)
            return fd

        def w_subdir(*a, **k):
            fd = real_subdir(*a, **k)
            opened.add(fd)
            return fd

        def w_lock(*a, **k):
            fd = real_lock(*a, **k)
            opened.add(fd)
            return fd

        def w_close(fd):
            closed.add(fd)
            real_close(fd)

        monkeypatch.setattr(pa, "_open_dir_fd", w_open_dir)
        monkeypatch.setattr(pa, "_open_or_create_subdir", w_subdir)
        monkeypatch.setattr(pa, "_open_lock_file", w_lock)
        monkeypatch.setattr(os, "close", w_close)
        return opened, closed

    def test_no_fd_leak_across_revisions_and_skills(self, tmp_path, monkeypatch):
        opened, closed = self._install_trackers(monkeypatch)
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        skills = [
            {
                "slug": f"skill-{i}",
                "files": [
                    {"path": "SKILL.md", "content": f"s{i}"},
                    {"path": f"sub/run-{i}.py", "content": "print(1)"},
                ],
            }
            for i in range(3)
        ]
        for rev in range(1, 6):  # 5 revisions, 3 multi-file skills each
            applier.apply(runtime_assignment(revision=rev, version_id=10 + rev, skills=skills))
        leaked = opened - closed
        assert not leaked, f"leaked fds: {leaked}"

    def test_no_fd_leak_on_revoked_apply(self, tmp_path, monkeypatch):
        opened, closed = self._install_trackers(monkeypatch)
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=1, version_id=19))
        applier.apply(
            runtime_assignment(revision=2, version_id=19, revoked=True, revoked_at="2026-07-12T10:00:00Z")
        )
        leaked = opened - closed
        assert not leaked, f"leaked fds: {leaked}"

    def test_fd_count_does_not_grow_across_many_applies(self, tmp_path):
        # Cross-check via /dev/fd (works on macOS and Linux): no fd growth.
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        skills = [
            {"slug": f"s{i}", "files": [{"path": "SKILL.md", "content": "x"}]} for i in range(4)
        ]
        applier.apply(runtime_assignment(revision=1, version_id=11, skills=skills))
        before = len(os.listdir("/dev/fd"))
        for rev in range(2, 12):  # 10 more applies
            applier.apply(runtime_assignment(revision=rev, version_id=10 + rev, skills=skills))
        after = len(os.listdir("/dev/fd"))
        # No sustained growth (allow a tiny transient slack).
        assert after <= before + 1, f"fd count grew: before={before} after={after}"


# --- security / path traversal ----------------------------------------------


class TestSecurity:
    def test_path_traversal_slug_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [{"slug": "../evil", "files": [{"path": "SKILL.md", "content": "x"}]}]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))
        assert not (profile / "managed").exists()

    def test_path_traversal_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [{"slug": "alpha", "files": [{"path": "../../etc/passwd", "content": "x"}]}]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))
        assert not (profile / "managed").exists()

    def test_absolute_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [{"slug": "alpha", "files": [{"path": "/etc/passwd", "content": "x"}]}]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))

    def test_backslash_file_path_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        skills = [{"slug": "alpha", "files": [{"path": "..\\evil", "content": "x"}]}]
        with pytest.raises(PathTraversalError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2, skills=skills))

    def test_checksum_mismatch_rejected_at_apply(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        good = runtime_assignment(revision=3, version_id=20)
        tampered = dataclasses.replace(good, manifest_sha256="0" * 64)
        with pytest.raises(ChecksumMismatchError):
            applier.apply(tampered)
        assert _read_assignment_json(profile)["revision"] == 2

    def test_symlink_managed_escaping_profile_rejected(self, tmp_path):
        profile = tmp_path / "profile"
        profile.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        os.symlink(outside, profile / "managed")
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))
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
        profile.mkdir(parents=True, exist_ok=True)
        os.symlink(outside, profile / ".managed-versions")
        with pytest.raises(ProfileTamperError):
            ProfileApplier(profile).apply(runtime_assignment(revision=2))
        assert list(outside.iterdir()) == []

    def test_lock_timeout_raises(self, tmp_path):
        profile = tmp_path / "profile"
        applier = ProfileApplier(profile)
        applier.apply(runtime_assignment(revision=2, version_id=19))
        # Hold the apply lock from a separate process for 3s.
        holder = subprocess.Popen(
            [sys.executable, str(_WORKER), "hold", str(profile), "3"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(_ROOT)},
        )
        assert json.loads(holder.stdout.readline())["ok"] is True
        with pytest.raises(ProfileLockTimeoutError):
            ProfileApplier(profile).apply(
                runtime_assignment(revision=3, version_id=20), lock_timeout=0.3
            )
        holder.wait(timeout=10)
