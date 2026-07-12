"""Atomic application of immutable assignments to a Hermes profile.

Ownership boundary (frozen by the MVP plan, Task 6):

* the applier owns ``<profile>/managed/`` and ``<profile>/skills/enterprise/``;
* it must not delete or overwrite ``sessions/``, ``memories/``,
  ``skills/personal/`` or ``skills/learned/``;
* it must not modify any other non-managed file.

Atomic update scheme
--------------------

Managed content lives in *versioned* real directories under
``<profile>/.managed-versions/<version_key>/``.  Two stable symlinks gate the
active version:

* ``<profile>/managed``  -> ``.managed-versions/<version_key>``
* ``<profile>/skills/enterprise`` -> ``../managed/skills/enterprise``

Because ``skills/enterprise`` resolves *through* ``managed``, a single atomic
swap of the ``managed`` symlink publishes the whole managed layer (assignment
metadata, policy, and enterprise skills) at once.

Application steps:

1. Verify the assignment's manifest/policy checksums (defense in depth).
2. Pre-flight: reject any pre-existing ``managed``, ``skills/enterprise``,
   ``.managed-versions`` or ``skills`` entry that is a real directory where a
   symlink is expected, or a symlink that resolves outside the profile.
3. Decide: initial / update / idempotent / stale / conflict / rollback using
   revision ordering.  Stale revisions are rejected; the same revision with
   different content fails closed; a higher revision pointing at an older
   ``versionId`` (rollback) is allowed.
4. Build the new version in a sibling staging directory under
   ``.managed-versions/.staging.<token>/``; write all files, fsync them and the
   directory, then re-verify the written bytes.
5. ``os.replace`` the staging directory into ``.managed-versions/<version_key>``
   (atomic, same filesystem), create a temporary ``managed`` symlink, and
   ``os.replace`` that symlink over ``managed`` (atomic publish).  If the
   publish fails the temp symlink and the new version dir are removed and the
   previous ``managed`` symlink — still pointing at the old version — is left
   intact.
6. Ensure the ``skills/enterprise`` tracking symlink exists.
7. Remove orphaned version directories from previous publishes.

Revocation writes a ``tombstone.json`` and an ``assignment.json`` with
``revoked=true``; it never deletes personal data.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional

from hermes_managed.contracts import (
    ChecksumMismatchError,
    RuntimeAssignment,
    canonical_json_bytes,
    sha256_hex,
    verify_manifest_checksum,
    verify_policy_checksum,
)

__all__ = [
    "ProfileStateError",
    "ProfileTamperError",
    "PathTraversalError",
    "StaleRevisionError",
    "RevisionConflictError",
    "ManagedState",
    "ApplyResult",
    "ProfileApplier",
]

# A module-level alias for os.replace so tests can simulate a publish failure.
_atomic_replace = os.replace

_MANAGED_LINK_NAME = "managed"
_VERSIONS_DIR_NAME = ".managed-versions"
_ENTERPRISE_LINK_REL = "skills/enterprise"
_ENTERPRISE_LINK_TARGET = "../managed/skills/enterprise"

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# --- exceptions ---------------------------------------------------------------


class ProfileStateError(Exception):
    """The profile's managed state is inconsistent or unexpected."""


class ProfileTamperError(ProfileStateError):
    """A managed path is a real directory or a symlink escaping the profile."""


class PathTraversalError(ProfileStateError):
    """An enterprise skill slug or file path is not a safe relative path."""


class StaleRevisionError(ProfileStateError):
    """The assignment revision is older than the currently applied revision."""


class RevisionConflictError(ProfileStateError):
    """The assignment revision matches the current one but the content differs."""


# --- state records ------------------------------------------------------------


@dataclass(frozen=True)
class ManagedState:
    """Read-only view of the currently applied managed state."""

    assignment_id: int
    revision: int
    template_id: int
    version_id: int
    manifest_sha256: str
    policy_sha256: str
    content_key: str
    revoked: bool
    revoked_at: Optional[str]
    version_dir: Path


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of :meth:`ProfileApplier.apply`."""

    revision: int
    version_id: int
    content_key: str
    status: str  # "initial" | "updated" | "idempotent" | "revoked"
    revoked: bool
    managed_path: Path
    version_dir: Path
    applied_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _silent_unlink(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _silent_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Not all platforms/filesystems support fsync on directory file
        # descriptors (e.g. macOS).  The file-level fsyncs already persist the
        # contents; this is best-effort for directory entry durability.
        pass
    finally:
        os.close(fd)


def _write_file_atomic(path: Path, content: str) -> None:
    """Write text to ``path`` and fsync it (parent must exist)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_file(path)


def _is_within(path: Path, base: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


def _validate_slug(slug: Any) -> str:
    if not isinstance(slug, str) or not _SLUG_RE.match(slug):
        raise PathTraversalError("enterprise skill slug is not a safe name")
    return slug


def _safe_skill_file_path(file_path: Any, base: Path) -> Path:
    if not isinstance(file_path, str) or not file_path:
        raise PathTraversalError("enterprise skill file path is empty")
    if "\\" in file_path:
        raise PathTraversalError("enterprise skill file path must not contain backslashes")
    if os.path.isabs(file_path):
        raise PathTraversalError("enterprise skill file path must be relative")
    parts = PurePosixPath(file_path).parts
    if any(p in ("", ".", "..") for p in parts):
        raise PathTraversalError("enterprise skill file path contains traversal segments")
    target = base / file_path
    if not _is_within(target, base):
        raise PathTraversalError("enterprise skill file path escapes the skill directory")
    return target


# --- applier ------------------------------------------------------------------


class ProfileApplier:
    """Applies immutable :class:`RuntimeAssignment` values to a profile."""

    def __init__(self, profile_dir: Path) -> None:
        self._profile = Path(profile_dir)

    @property
    def profile_dir(self) -> Path:
        return self._profile

    # -- public API --

    def apply(
        self,
        assignment: RuntimeAssignment,
        *,
        clock: Optional[Callable[[], str]] = None,
    ) -> ApplyResult:
        """Atomically apply ``assignment`` to the profile."""
        # 1. Defense-in-depth checksum verification (parser already checked).
        verify_manifest_checksum(assignment.manifest, assignment.manifest_sha256)
        verify_policy_checksum(assignment.effective_model_policy)

        # 2. Pre-flight safety: refuse tampered / escaping paths before writing.
        self._preflight_safety()

        # 3. Decide what to do relative to the current managed state.
        current = self._read_current()
        self._enforce_revision_semantics(current, assignment)
        content_key = self._content_key(assignment)

        if current is not None and current.content_key == content_key:
            # Idempotent re-apply of identical content: no-op.
            return ApplyResult(
                revision=assignment.revision,
                version_id=assignment.version_id,
                content_key=content_key,
                status="revoked" if assignment.revoked else "idempotent",
                revoked=assignment.revoked,
                managed_path=self._managed_link(),
                version_dir=current.version_dir,
                applied_at=_read_applied_at(current.version_dir),
            )

        # 4-5. Materialize into staging and publish atomically.
        version_key = self._version_key(assignment, content_key)
        staging = self._versions_dir() / f".staging.{secrets.token_hex(8)}"
        applied_at = (clock or _now_iso)()
        try:
            self._materialize(staging, assignment, content_key, version_key, applied_at)
            self._publish(staging, version_key)
        except Exception:
            _silent_rmtree(staging)
            raise

        # 6. Ensure the enterprise tracking symlink exists.
        self._ensure_enterprise_symlink()

        # 7. Clean orphaned version directories.
        self._cleanup_old_versions(keep=version_key)

        version_dir = self._versions_dir() / version_key
        return ApplyResult(
            revision=assignment.revision,
            version_id=assignment.version_id,
            content_key=content_key,
            status="revoked" if assignment.revoked else (
                "initial" if current is None else "updated"
            ),
            revoked=assignment.revoked,
            managed_path=self._managed_link(),
            version_dir=version_dir,
            applied_at=applied_at,
        )

    @classmethod
    def read_state(cls, profile_dir: Path) -> Optional[ManagedState]:
        """Return the currently applied managed state, or ``None`` if unmanaged."""
        applier = cls(profile_dir)
        return applier._read_current()

    @classmethod
    def is_enabled(cls, profile_dir: Path) -> bool:
        """True iff a non-revoked managed assignment is currently applied."""
        state = cls.read_state(profile_dir)
        return state is not None and not state.revoked

    # -- path helpers --

    def _managed_link(self) -> Path:
        return self._profile / _MANAGED_LINK_NAME

    def _versions_dir(self) -> Path:
        return self._profile / _VERSIONS_DIR_NAME

    def _enterprise_link(self) -> Path:
        return self._profile / _ENTERPRISE_LINK_REL

    # -- pre-flight --

    def _preflight_safety(self) -> None:
        profile = self._profile.resolve()
        self._profile.mkdir(parents=True, exist_ok=True)

        versions = self._versions_dir()
        if versions.is_symlink():
            raise ProfileTamperError(".managed-versions must not be a symlink")
        if versions.exists() and not versions.is_dir():
            raise ProfileTamperError(".managed-versions must be a directory")

        managed = self._managed_link()
        if managed.is_symlink():
            if not managed.exists():
                raise ProfileTamperError("managed symlink is broken")
            if not _is_within(managed.resolve(), profile):
                raise ProfileTamperError("managed symlink resolves outside the profile")
        elif managed.exists():
            raise ProfileTamperError("managed must be a symlink, not a real directory")

        skills = self._profile / "skills"
        if skills.is_symlink():
            if not _is_within(skills.resolve(), profile):
                raise ProfileTamperError("skills symlink resolves outside the profile")
        elif skills.exists() and not skills.is_dir():
            raise ProfileTamperError("skills must be a directory")

        enterprise = self._enterprise_link()
        if enterprise.is_symlink():
            if not enterprise.exists():
                raise ProfileTamperError("skills/enterprise symlink is broken")
            if not _is_within(enterprise.resolve(), profile):
                raise ProfileTamperError("skills/enterprise symlink resolves outside the profile")
        elif enterprise.exists():
            raise ProfileTamperError(
                "skills/enterprise must be a symlink, not a real directory"
            )

    # -- current state --

    def _read_current(self) -> Optional[ManagedState]:
        managed = self._managed_link()
        if not managed.is_symlink():
            return None
        if not managed.exists():
            raise ProfileTamperError("managed symlink is broken")
        version_dir = managed.resolve()
        record_path = version_dir / "assignment.json"
        if not record_path.exists():
            return None
        record = json.loads(record_path.read_text())
        return ManagedState(
            assignment_id=record["assignment_id"],
            revision=record["revision"],
            template_id=record["template_id"],
            version_id=record["version_id"],
            manifest_sha256=record["manifest_sha256"],
            policy_sha256=record["policy_sha256"],
            content_key=record["content_key"],
            revoked=record["revoked"],
            revoked_at=record.get("revoked_at"),
            version_dir=version_dir,
        )

    # -- revision semantics --

    def _enforce_revision_semantics(
        self, current: Optional[ManagedState], assignment: RuntimeAssignment
    ) -> None:
        if current is None:
            return
        if assignment.revision < current.revision:
            raise StaleRevisionError(
                f"stale revision {assignment.revision} < current {current.revision}"
            )
        if assignment.revision == current.revision:
            new_key = self._content_key(assignment)
            if new_key != current.content_key:
                raise RevisionConflictError(
                    f"revision {assignment.revision} already applied with different content"
                )
        # assignment.revision > current.revision -> update / rollback, allowed.

    # -- content addressing --

    @staticmethod
    def _content_key(assignment: RuntimeAssignment) -> str:
        policy = assignment.effective_model_policy
        policy_sha = policy.get("policySha256", "")
        record = {
            "assignment_id": assignment.assignment_id,
            "revision": assignment.revision,
            "template_id": assignment.template_id,
            "version_id": assignment.version_id,
            "manifest_sha256": assignment.manifest_sha256,
            "policy_sha256": policy_sha,
            "revoked": assignment.revoked,
            "revoked_at": assignment.revoked_at,
        }
        return sha256_hex(canonical_json_bytes(record))

    @staticmethod
    def _version_key(assignment: RuntimeAssignment, content_key: str) -> str:
        return f"r{assignment.revision:020d}-{content_key[:24]}"

    # -- materialization --

    def _materialize(
        self,
        staging: Path,
        assignment: RuntimeAssignment,
        content_key: str,
        version_key: str,
        applied_at: str,
    ) -> None:
        staging.mkdir(parents=True, exist_ok=False)
        policy = dict(assignment.effective_model_policy)
        policy_sha = policy.get("policySha256", "")

        record = {
            "assignment_id": assignment.assignment_id,
            "revision": assignment.revision,
            "template_id": assignment.template_id,
            "version_id": assignment.version_id,
            "template_slug": assignment.template_slug,
            "display_name": assignment.display_name,
            "manifest_sha256": assignment.manifest_sha256,
            "policy_sha256": policy_sha,
            "content_key": content_key,
            "version_key": version_key,
            "revoked": assignment.revoked,
            "revoked_at": assignment.revoked_at,
            "applied_at": applied_at,
        }

        _write_file_atomic(staging / "assignment.json", canonical_json(record))
        _write_file_atomic(staging / "manifest.json", canonical_json(dict(assignment.manifest)))
        _write_file_atomic(staging / "policy.json", canonical_json(policy))

        enterprise_dir = staging / "skills" / "enterprise"
        enterprise_dir.mkdir(parents=True, exist_ok=True)

        if assignment.revoked:
            tombstone = {
                "revoked": True,
                "revoked_at": assignment.revoked_at,
                "assignment_id": assignment.assignment_id,
                "revision": assignment.revision,
                "version_id": assignment.version_id,
            }
            _write_file_atomic(staging / "tombstone.json", canonical_json(tombstone))
        else:
            self._materialize_enterprise_skills(assignment, enterprise_dir)

        _fsync_dir(staging)
        _fsync_dir(staging / "skills")
        _fsync_dir(enterprise_dir)

        # Verify the written bytes match the signed checksums.
        self._verify_staging(staging, assignment)

    def _materialize_enterprise_skills(
        self, assignment: RuntimeAssignment, enterprise_dir: Path
    ) -> None:
        manifest = assignment.manifest
        skills = manifest.get("enterpriseSkills", []) if isinstance(manifest, Mapping) else []
        if skills is None:
            skills = []
        if not isinstance(skills, list):
            raise ProfileStateError("manifest 'enterpriseSkills' must be a list")
        for skill in skills:
            if not isinstance(skill, Mapping):
                raise ProfileStateError("each enterprise skill must be an object")
            slug = _validate_slug(skill.get("slug"))
            skill_dir = enterprise_dir / slug
            skill_dir.mkdir(parents=True, exist_ok=False)
            files = skill.get("files", [])
            if not isinstance(files, list):
                raise ProfileStateError("enterprise skill 'files' must be a list")
            for entry in files:
                if not isinstance(entry, Mapping):
                    raise ProfileStateError("each enterprise skill file must be an object")
                target = _safe_skill_file_path(entry.get("path"), skill_dir)
                content = entry.get("content")
                if not isinstance(content, str):
                    raise ProfileStateError("enterprise skill file 'content' must be a string")
                _write_file_atomic(target, content)

    def _verify_staging(self, staging: Path, assignment: RuntimeAssignment) -> None:
        manifest_bytes = (staging / "manifest.json").read_bytes()
        policy_bytes = (staging / "policy.json").read_bytes()
        manifest_obj = json.loads(manifest_bytes)
        policy_obj = json.loads(policy_bytes)
        if sha256_hex(manifest_bytes) != sha256_hex(canonical_json_bytes(assignment.manifest)):
            raise ChecksumMismatchError("staged manifest bytes do not match the signed digest")
        # policySha256 is verified by verify_policy_checksum on the staged copy.
        verify_policy_checksum(policy_obj)
        # Confirm the manifest object round-trips to the signed digest.
        verify_manifest_checksum(manifest_obj, assignment.manifest_sha256)

    # -- publish --

    def _publish(self, staging: Path, version_key: str) -> None:
        versions_dir = self._versions_dir()
        versions_dir.mkdir(parents=True, exist_ok=True)
        version_dir = versions_dir / version_key
        if version_dir.exists():
            raise ProfileStateError(f"version directory already exists: {version_key}")

        # Atomic rename of the staging directory into its final name.
        os.replace(staging, version_dir)
        _fsync_dir(versions_dir)

        # Create a temporary symlink, then atomically swap it onto `managed`.
        temp_link = self._profile / f".managed.swap.{secrets.token_hex(8)}"
        relative_target = f"{_VERSIONS_DIR_NAME}/{version_key}"
        try:
            os.symlink(relative_target, temp_link)
        except OSError:
            _silent_unlink(temp_link)
            _silent_rmtree(version_dir)
            raise
        try:
            _atomic_replace(temp_link, self._managed_link())
        except OSError:
            _silent_unlink(temp_link)
            _silent_rmtree(version_dir)
            raise
        _fsync_dir(self._profile)

    def _ensure_enterprise_symlink(self) -> None:
        skills_dir = self._profile / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        enterprise = self._enterprise_link()
        if enterprise.is_symlink():
            # Already tracks `managed`; validated in pre-flight.
            return
        if enterprise.exists():
            # Pre-flight would have raised for a real directory; be defensive.
            raise ProfileTamperError("skills/enterprise must be a symlink, not a real directory")
        os.symlink(_ENTERPRISE_LINK_TARGET, enterprise)
        _fsync_dir(skills_dir)

    def _cleanup_old_versions(self, *, keep: str) -> None:
        versions_dir = self._versions_dir()
        if not versions_dir.is_dir():
            return
        active = self._managed_link().resolve()
        for entry in versions_dir.iterdir():
            if entry.name == keep:
                continue
            if entry.name.startswith(".staging.") or entry.name.startswith(".managed.swap."):
                _silent_rmtree(entry) if entry.is_dir() else _silent_unlink(entry)
                continue
            # Keep the directory that `managed` currently resolves to (it may
            # differ from `keep` briefly during recovery); otherwise remove
            # orphaned version dirs.
            try:
                if entry.resolve() == active:
                    continue
            except OSError:
                pass
            if entry.is_dir():
                _silent_rmtree(entry)


def _read_applied_at(version_dir: Path) -> str:
    try:
        record = json.loads((version_dir / "assignment.json").read_text())
        return record.get("applied_at", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def canonical_json(obj: Any) -> str:
    """Canonical JSON text for managed-layer files (matches contracts.algorithm)."""
    return canonical_json_bytes(obj).decode("utf-8")
