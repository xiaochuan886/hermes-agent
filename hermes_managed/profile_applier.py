"""Atomic application of immutable assignments to a Hermes profile.

Ownership boundary (frozen by the MVP plan, Task 6):

* the applier owns ``<profile>/managed/`` and ``<profile>/skills/enterprise/``;
* it must not delete or overwrite ``sessions/``, ``memories/``,
  ``skills/personal/`` or ``skills/learned/``;
* it must not modify any other non-managed file.

Atomic update scheme
--------------------

Managed content lives in versioned real directories under
``<profile>/.managed-versions/<version_key>/``.  Two stable symlinks gate the
active version:

* ``<profile>/managed``  -> ``.managed-versions/<version_key>``
* ``<profile>/skills/enterprise`` -> ``../managed/skills/enterprise``

Because ``skills/enterprise`` resolves *through* ``managed``, swapping the
``managed`` symlink publishes the whole managed layer at once.

Single commit point
-------------------

All fallible preparation happens before the unique commit point — the
``os.replace`` that swaps the ``managed`` symlink:

1. acquire the cross-process apply lock (covers read → decide → materialize →
   publish → active-state confirm);
2. open trusted directory file descriptors (``profile``, ``.managed-versions``,
   ``skills``) with ``O_DIRECTORY | O_NOFOLLOW`` and operate relative to them
   so a check-then-replace (TOCTOU) of those entries cannot redirect writes;
3. verify checksums, decide revision semantics, materialize a sibling staging
   version dir (fsync every file + directory, re-verify the written bytes);
4. rename staging → ``.managed-versions/<version_key>`` (fsync the versions
   dir);
5. prepare the ``skills/enterprise`` tracking symlink (create or reclaim it
   atomically; fsync the skills dir);
6. re-verify that ``.managed-versions`` and ``skills`` entries still pin the
   same real directories the open fds refer to (defeats mid-apply replacement);
7. **commit**: ``os.replace`` a fresh ``managed`` symlink into place (fsync the
   profile dir).

Steps 1–6 are reversible: on any failure the staging dir, the new version dir
and (on first apply) the dangling enterprise link are removed and the previous
``managed`` symlink is left intact.  Step 7 is atomic; if it fails the previous
state is still active.  After the commit, only best-effort cleanup of orphaned
version directories runs — its failure never makes a successful apply report
failure.

TOCTOU / symlink safety
-----------------------

* every managed root is opened with ``O_NOFOLLOW`` (refuses symlinks) and held
  as a directory fd; subsequent operations are relative to that fd, so replacing
  the directory *entry* with a symlink cannot redirect writes (the fd still pins
  the original real directory);
* ``.managed-versions`` and ``skills`` must be real directories in managed mode;
* the current state is read via ``readlink(managed)`` + validation + opening the
  version dir through the trusted ``versions_fd`` (never by following the
  ``managed`` symlink);
* cleanup uses ``lstat`` (no follow) on every entry, skips symlinks entirely and
  re-confirms each entry lives under the pinned ``versions_fd`` — a symlink
  planted in ``.managed-versions`` is never followed and its target is never
  deleted.

macOS note: ``fsync`` on a directory file descriptor returns ``EINVAL`` (not
supported); :func:`_fsync_dir_fd` treats that as a tolerated degradation and
only surfaces genuine errors.  File-level ``fsync`` is always honored.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional

from hermes_managed.contracts import (
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
    "ProfileLockTimeoutError",
    "ProfileRollbackError",
    "ProfileDurabilityUncertainError",
    "ManagedState",
    "ApplyResult",
    "ProfileApplier",
]

# Module-level alias for os.replace so tests can simulate a commit failure.
_atomic_replace = os.replace

_MANAGED_LINK_NAME = "managed"
_VERSIONS_DIR_NAME = ".managed-versions"
_LOCK_FILE_NAME = ".apply.lock"
_ENTERPRISE_LINK_NAME = "enterprise"
_ENTERPRISE_LINK_TARGET = "../managed/skills/enterprise"

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


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


class ProfileLockTimeoutError(ProfileStateError):
    """The cross-process apply lock could not be acquired in time."""


class ProfileRollbackError(ProfileStateError):
    """Rollback after a failed commit itself failed (state left for inspection)."""


class ProfileDurabilityUncertainError(ProfileStateError):
    """The new managed state was published but a post-commit durability sync
    (directory fsync) failed.

    The new revision IS active — ``managed`` already points at the new version
    directory.  This error explicitly does NOT claim the previous state is still
    valid.  The caller MUST NOT treat this as "old state preserved"; retrying the
    same revision/content is safe and idempotent (the new version dir is intact).
    """


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
    version_key: str
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


# --- low-level fd helpers -----------------------------------------------------


def _open_dir_fd(path: Any, dir_fd: Optional[int] = None) -> int:
    """Open a directory with O_DIRECTORY | O_NOFOLLOW (refuses symlinks)."""
    return os.open(str(path), os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=dir_fd)


def _open_dir_fd_retry(name: str, dir_fd: int, attempts: int = 12, delay: float = 0.002) -> int:
    """Open a directory relative to ``dir_fd`` with a bounded retry.

    macOS exhibits a rare kernel race where ``openat`` on a directory fd
    concurrently creating entries transiently returns ``ENOENT`` even though
    the entry exists.  This affects only fixed-name managed roots
    (``.managed-versions``, ``skills``) when two processes initialize the same
    profile at once.  Unique names (staging / version dirs) never race.  A
    tampered symlink is detected via ``ENOTDIR``/``ELOOP`` (not retried).
    """
    last: Optional[FileNotFoundError] = None
    for _ in range(attempts):
        try:
            return _open_dir_fd(name, dir_fd=dir_fd)
        except FileNotFoundError as exc:
            last = exc
            time.sleep(delay)
    assert last is not None
    raise last


def _open_lock_file(versions_fd: int) -> int:
    """Open/create the apply lock relative to ``versions_fd`` (retry on ENOENT)."""
    last: Optional[FileNotFoundError] = None
    for _ in range(12):
        try:
            return os.open(
                _LOCK_FILE_NAME,
                os.O_RDWR | os.O_CREAT | _NOFOLLOW,
                0o600,
                dir_fd=versions_fd,
            )
        except FileNotFoundError as exc:
            last = exc
            time.sleep(0.002)
    assert last is not None
    raise last


def _open_or_create_subdir(parent_fd: int, name: str) -> int:
    """Open subdir ``name`` relative to ``parent_fd``; create it if absent."""
    try:
        return _open_dir_fd(name, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, dir_fd=parent_fd)
            _fsync_dir_fd(parent_fd)
        except FileExistsError:
            pass  # another process created it concurrently
        return _open_dir_fd_retry(name, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            # O_NOFOLLOW|O_DIRECTORY on a symlink: ELOOP (Linux) or ENOTDIR (macOS).
            raise ProfileTamperError(f"{name} must not be a symlink") from exc
        raise


def _fsync_dir_fd(fd: int) -> None:
    """Best-effort fsync of a directory fd.

    macOS returns ``EINVAL`` for fsync on a directory fd (unsupported); that is
    a tolerated degradation.  Any other error is surfaced so genuine durability
    failures (and injected faults) are not swallowed.
    """
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno == errno.EINVAL:
            return
        raise


def _fsync_file_fd(fd: int) -> None:
    os.fsync(fd)


def _read_fd(fd: int) -> str:
    with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as fh:
        return fh.read()


def _lstat_rel(name: str, dir_fd: int) -> os.stat_result:
    return os.lstat(name, dir_fd=dir_fd)


def _unlink_rel(name: str, dir_fd: int) -> None:
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        pass


def _rmtree_fd(parent_fd: int, name: str) -> None:
    """Recursively remove ``name`` (a real dir) relative to ``parent_fd``.

    Never follows symlinks: each entry is ``lstat``-ed; a symlink entry is
    unlinked (its target untouched), a dir entry is recursed into via a fresh
    ``O_NOFOLLOW`` fd.
    """
    child_fd = _open_dir_fd(name, dir_fd=parent_fd)
    try:
        for entry in os.listdir(child_fd):
            st = os.lstat(entry, dir_fd=child_fd)
            if stat.S_ISLNK(st.st_mode):
                os.unlink(entry, dir_fd=child_fd)
            elif stat.S_ISDIR(st.st_mode):
                _rmtree_fd(child_fd, entry)
            else:
                os.unlink(entry, dir_fd=child_fd)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _remove_entry(parent_fd: int, name: str) -> None:
    """Remove one entry under ``parent_fd`` without following symlinks.

    A symlink entry is unlinked (target preserved).  A regular file is unlinked.
    A directory is recursively removed via :func:`_rmtree_fd`.
    """
    st = os.lstat(name, dir_fd=parent_fd)
    if stat.S_ISLNK(st.st_mode):
        os.unlink(name, dir_fd=parent_fd)
    elif stat.S_ISDIR(st.st_mode):
        _rmtree_fd(parent_fd, name)
    else:
        os.unlink(name, dir_fd=parent_fd)


def _write_rel(base_fd: int, relpath: str, content: str) -> None:
    """Write ``content`` to ``relpath`` relative to ``base_fd`` (atomic + fsync).

    Intermediate directories are created as needed.  ``O_NOFOLLOW`` is used on
    every open so a symlink trap cannot redirect the write.
    """
    parts = PurePosixPath(relpath).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise PathTraversalError("managed file path contains traversal segments")
    if any("\\" in p for p in parts):
        raise PathTraversalError("managed file path must not contain backslashes")

    cur_fd = base_fd
    opened: list[int] = []
    try:
        for part in parts[:-1]:
            cur_fd = _open_or_create_subdir(cur_fd, part)
            opened.append(cur_fd)
        fname = parts[-1]
        tmp = fname + ".tmp"
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _NOFOLLOW, 0o600, dir_fd=cur_fd
        )
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as fh:
            fh.write(content)
            fh.flush()
            _fsync_file_fd(fh.fileno())
        _atomic_replace(tmp, fname, src_dir_fd=cur_fd, dst_dir_fd=cur_fd)
        _fsync_dir_fd(cur_fd)
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _read_rel(base_fd: int, relpath: str) -> str:
    parts = PurePosixPath(relpath).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise PathTraversalError("managed file path contains traversal segments")
    cur_fd = base_fd
    opened: list[int] = []
    try:
        for part in parts[:-1]:
            cur_fd = _open_dir_fd(part, dir_fd=cur_fd)
            opened.append(cur_fd)
        fd = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW, dir_fd=cur_fd)
        try:
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as fh:
                return fh.read()
        finally:
            os.close(fd)
    finally:
        for fd in reversed(opened):
            os.close(fd)


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


def _safe_skill_file_path(file_path: Any) -> str:
    if not isinstance(file_path, str) or not file_path:
        raise PathTraversalError("enterprise skill file path is empty")
    if "\\" in file_path:
        raise PathTraversalError("enterprise skill file path must not contain backslashes")
    if os.path.isabs(file_path):
        raise PathTraversalError("enterprise skill file path must be relative")
    parts = PurePosixPath(file_path).parts
    if any(p in ("", ".", "..") for p in parts):
        raise PathTraversalError("enterprise skill file path contains traversal segments")
    return file_path


def _read_applied_at(profile: Path, version_key: str) -> str:
    try:
        record = json.loads(
            (profile / _VERSIONS_DIR_NAME / version_key / "assignment.json").read_text()
        )
        return record.get("applied_at", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _symlink_rel_fd(target: str, name: str, parent_fd: int, parent_path: Path) -> None:
    """Create a symlink ``name`` -> ``target`` inside the dir pinned by ``parent_fd``.

    ``os.symlink`` does not accept ``dst_dir_fd`` on macOS (and ``/dev/fd/<fd>``
    is not traversable there), so this helper prefers the fd-relative form where
    available (Linux) and falls back to creating via the parent *path* followed
    by an immediate ``lstat`` relative to ``parent_fd``.  The post-create verify
    confirms the symlink landed in the pinned directory: if the parent entry was
    swapped to a symlink mid-apply, the ``lstat`` (fd-relative) does not find the
    new entry and raises — fail closed.  Residual platform limit: on the macOS
    fallback path there is a small window between path-based creation and the
    verify where a swapped parent receives a dangling temp symlink entry (no
    file content is written outside the profile).
    """
    try:
        os.symlink(target, name, dst_dir_fd=parent_fd)
        return
    except TypeError:
        pass  # macOS: dst_dir_fd unsupported
    os.symlink(target, os.path.join(str(parent_path), name))
    try:
        st = os.lstat(name, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise ProfileTamperError(
            "symlink creation did not land in the pinned parent directory"
        ) from exc
    if not stat.S_ISLNK(st.st_mode):
        raise ProfileTamperError("created managed entry is not a symlink")


def canonical_json(obj: Any) -> str:
    """Canonical JSON text for managed-layer files."""
    return canonical_json_bytes(obj).decode("utf-8")


# --- applier ------------------------------------------------------------------


class ProfileApplier:
    """Applies immutable :class:`RuntimeAssignment` values to a profile."""

    def __init__(self, profile_dir: Path) -> None:
        self._profile = Path(profile_dir)
        # Test seams (not public API); default to no-ops.
        self._after_open_hook: Optional[Callable[[], None]] = None
        self._pre_commit_hook: Optional[Callable[[], None]] = None

    @property
    def profile_dir(self) -> Path:
        return self._profile

    # -- public API --

    def apply(
        self,
        assignment: RuntimeAssignment,
        *,
        clock: Optional[Callable[[], str]] = None,
        lock_timeout: float = 30.0,
        lock_sleep: Callable[[float], None] = time.sleep,
    ) -> ApplyResult:
        """Atomically apply ``assignment`` to the profile under an exclusive lock."""
        verify_manifest_checksum(assignment.manifest, assignment.manifest_sha256)
        verify_policy_checksum(assignment.effective_model_policy)

        profile_fd, versions_fd, lock_fd = self._open_and_lock(
            exclusive=True, timeout=lock_timeout, sleep=lock_sleep
        )
        skills_fd: Optional[int] = None
        try:
            if self._after_open_hook is not None:
                self._after_open_hook()

            skills_fd = self._open_skills_dir(profile_fd)
            current = self._read_current(profile_fd, versions_fd)
            self._enforce_revision_semantics(current, assignment)
            content_key = self._content_key(assignment)

            if current is not None and current.content_key == content_key:
                return ApplyResult(
                    revision=assignment.revision,
                    version_id=assignment.version_id,
                    content_key=content_key,
                    status="revoked" if assignment.revoked else "idempotent",
                    revoked=assignment.revoked,
                    managed_path=self._profile / _MANAGED_LINK_NAME,
                    version_dir=self._profile / _VERSIONS_DIR_NAME / current.version_key,
                    applied_at=_read_applied_at(self._profile, current.version_key),
                )

            version_key = self._version_key(assignment, content_key)
            applied_at = (clock or _now_iso)()
            first_apply = current is None
            committed = False
            staging_name: Optional[str] = None
            try:
                staging_name = self._materialize(
                    versions_fd, assignment, content_key, version_key, applied_at
                )
                self._rename_to_version(versions_fd, staging_name, version_key)
                staging_name = None  # consumed by the rename
                self._prepare_enterprise_link(skills_fd)
                self._verify_entries_unchanged(profile_fd, versions_fd, skills_fd)
                if self._pre_commit_hook is not None:
                    self._pre_commit_hook()
                # Commit point: atomically swap the managed symlink.  Nothing
                # fallible may run between this succeeding and ``committed=True``.
                self._commit_managed_link(profile_fd, version_key)
                committed = True
                # Post-commit durability sync.  If this fails the new state is
                # already published — it MUST NOT be rolled back.
                self._fsync_profile_dir(profile_fd)
            except BaseException as exc:
                if not committed:
                    # Pre-commit failure: rollback preparatory side effects and
                    # preserve the previous active state.
                    rollback_error: Optional[BaseException] = None
                    try:
                        self._rollback(
                            profile_fd, versions_fd, skills_fd, version_key,
                            staging_name, first_apply,
                        )
                    except Exception as rerr:
                        rollback_error = rerr
                    if rollback_error is not None:
                        raise ProfileRollbackError(
                            "rollback failed after a failed commit"
                        ) from rollback_error
                    raise
                # committed == True: post-commit durability failure.  The new
                # revision is already active — do NOT rollback (that would
                # delete the active version dir) and do NOT run cleanup.  Surface
                # as durability-uncertain; retrying the same revision is safe
                # and idempotent because the new version dir is intact.
                raise ProfileDurabilityUncertainError(
                    "new managed state was published but profile directory "
                    "durability could not be confirmed; the new revision is active"
                ) from exc

            # Post-commit best-effort cleanup.  Its failure must NOT fail apply.
            # Only reached when the durability sync succeeded.
            try:
                self._cleanup_old_versions(versions_fd, keep=version_key)
            except Exception:
                pass

            return ApplyResult(
                revision=assignment.revision,
                version_id=assignment.version_id,
                content_key=content_key,
                status="revoked" if assignment.revoked else (
                    "initial" if first_apply else "updated"
                ),
                revoked=assignment.revoked,
                managed_path=self._profile / _MANAGED_LINK_NAME,
                version_dir=self._profile / _VERSIONS_DIR_NAME / version_key,
                applied_at=applied_at,
            )
        finally:
            if skills_fd is not None:
                os.close(skills_fd)
            os.close(versions_fd)
            os.close(profile_fd)
            os.close(lock_fd)

    @classmethod
    def read_state(cls, profile_dir: Path) -> Optional[ManagedState]:
        """Return the currently applied managed state, or ``None`` if unmanaged."""
        applier = cls(profile_dir)
        return applier._read_state_locked()

    @classmethod
    def is_enabled(cls, profile_dir: Path) -> bool:
        """True iff a non-revoked managed assignment is currently applied."""
        state = cls.read_state(profile_dir)
        return state is not None and not state.revoked

    # -- locking & trusted dirs --

    def _open_and_lock(
        self, *, exclusive: bool, timeout: float, sleep: Callable[[float], None]
    ) -> tuple[int, int, int]:
        self._profile.mkdir(parents=True, exist_ok=True)
        profile_fd = _open_dir_fd(self._profile)
        try:
            versions_fd = self._open_or_create_versions(profile_fd)
            lock_fd = _open_lock_file(versions_fd)
        except BaseException:
            os.close(profile_fd)
            raise
        try:
            self._flock(lock_fd, exclusive=exclusive, timeout=timeout, sleep=sleep)
        except BaseException:
            os.close(lock_fd)
            os.close(versions_fd)
            os.close(profile_fd)
            raise
        return profile_fd, versions_fd, lock_fd

    def _open_or_create_versions(self, profile_fd: int) -> int:
        try:
            return _open_dir_fd(_VERSIONS_DIR_NAME, dir_fd=profile_fd)
        except FileNotFoundError:
            try:
                os.mkdir(_VERSIONS_DIR_NAME, dir_fd=profile_fd)
                _fsync_dir_fd(profile_fd)
            except FileExistsError:
                pass  # another process created it concurrently
            return _open_dir_fd_retry(_VERSIONS_DIR_NAME, dir_fd=profile_fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ProfileTamperError(
                    ".managed-versions must not be a symlink"
                ) from exc
            raise

    def _open_skills_dir(self, profile_fd: int) -> int:
        try:
            return _open_or_create_subdir(profile_fd, "skills")
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise ProfileTamperError("skills must not be a symlink") from exc
            raise

    def _flock(
        self, lock_fd: int, *, exclusive: bool, timeout: float, sleep: Callable[[float], None]
    ) -> None:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fd, op | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProfileLockTimeoutError(
                        f"could not acquire apply lock within {timeout}s"
                    )
                sleep(min(0.05, max(0.0, remaining)))

    def _read_state_locked(self) -> Optional[ManagedState]:
        if not (self._profile / _VERSIONS_DIR_NAME).exists():
            return None
        profile_fd, versions_fd, lock_fd = self._open_and_lock(
            exclusive=False, timeout=30.0, sleep=time.sleep
        )
        try:
            return self._read_current(profile_fd, versions_fd)
        finally:
            os.close(versions_fd)
            os.close(profile_fd)
            os.close(lock_fd)

    # -- current state (TOCTOU-safe: readlink + validate + open via versions_fd) --

    def _read_current(self, profile_fd: int, versions_fd: int) -> Optional[ManagedState]:
        try:
            target = os.readlink(_MANAGED_LINK_NAME, dir_fd=profile_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno == errno.EINVAL:
                raise ProfileTamperError(
                    "managed must be a symlink, not a real directory"
                ) from exc
            raise

        prefix = _VERSIONS_DIR_NAME + "/"
        if not target.startswith(prefix):
            raise ProfileTamperError("managed symlink target is unexpected")
        version_key = target[len(prefix):]
        if "/" in version_key or version_key.startswith("."):
            raise ProfileTamperError("managed symlink target is unexpected")

        try:
            raw = _read_rel(versions_fd, f"{version_key}/assignment.json")
        except FileNotFoundError:
            return None
        record = json.loads(raw)
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
            version_key=version_key,
            version_dir=self._profile / _VERSIONS_DIR_NAME / version_key,
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
        versions_fd: int,
        assignment: RuntimeAssignment,
        content_key: str,
        version_key: str,
        applied_at: str,
    ) -> str:
        staging_name = f".staging.{secrets.token_hex(8)}"
        os.mkdir(staging_name, dir_fd=versions_fd)
        staging_fd = _open_dir_fd(staging_name, dir_fd=versions_fd)
        try:
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
            _write_rel(staging_fd, "assignment.json", canonical_json(record))
            _write_rel(staging_fd, "manifest.json", canonical_json(dict(assignment.manifest)))
            _write_rel(staging_fd, "policy.json", canonical_json(policy))

            # skills/enterprise/ always exists in the version dir so the
            # tracking symlink is never dangling after the commit.
            os.mkdir("skills", dir_fd=staging_fd)
            skills_in_staging_fd = _open_dir_fd("skills", dir_fd=staging_fd)
            try:
                os.mkdir("enterprise", dir_fd=skills_in_staging_fd)
                ent_fd = _open_dir_fd("enterprise", dir_fd=skills_in_staging_fd)
                try:
                    if assignment.revoked:
                        tombstone = {
                            "revoked": True,
                            "revoked_at": assignment.revoked_at,
                            "assignment_id": assignment.assignment_id,
                            "revision": assignment.revision,
                            "version_id": assignment.version_id,
                        }
                        _write_rel(staging_fd, "tombstone.json", canonical_json(tombstone))
                    else:
                        self._materialize_enterprise_skills(assignment, ent_fd)
                finally:
                    os.close(ent_fd)
            finally:
                os.close(skills_in_staging_fd)

            _fsync_dir_fd(staging_fd)
            self._verify_staging(staging_fd, assignment)
        finally:
            os.close(staging_fd)
        return staging_name

    def _materialize_enterprise_skills(
        self, assignment: RuntimeAssignment, enterprise_fd: int
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
            # Ensure the skill directory exists (even when it has no files).
            # The returned fd is closed immediately — ownership: opener closes.
            # (_write_rel below re-opens/creates the dir as needed and closes
            # its own fd.)
            slug_fd = _open_or_create_subdir(enterprise_fd, slug)
            os.close(slug_fd)
            files = skill.get("files", [])
            if not isinstance(files, list):
                raise ProfileStateError("enterprise skill 'files' must be a list")
            for entry in files:
                if not isinstance(entry, Mapping):
                    raise ProfileStateError("each enterprise skill file must be an object")
                rel = _safe_skill_file_path(entry.get("path"))
                content = entry.get("content")
                if not isinstance(content, str):
                    raise ProfileStateError("enterprise skill file 'content' must be a string")
                _write_rel(enterprise_fd, f"{slug}/{rel}", content)

    def _verify_staging(self, staging_fd: int, assignment: RuntimeAssignment) -> None:
        manifest_obj = json.loads(_read_rel(staging_fd, "manifest.json"))
        policy_obj = json.loads(_read_rel(staging_fd, "policy.json"))
        verify_manifest_checksum(manifest_obj, assignment.manifest_sha256)
        verify_policy_checksum(policy_obj)

    # -- publish helpers --

    def _rename_to_version(
        self, versions_fd: int, staging_name: str, version_key: str
    ) -> None:
        try:
            _lstat_rel(version_key, versions_fd)
            raise ProfileStateError(f"version directory already exists: {version_key}")
        except FileNotFoundError:
            pass
        _atomic_replace(
            staging_name, version_key, src_dir_fd=versions_fd, dst_dir_fd=versions_fd
        )
        _fsync_dir_fd(versions_fd)

    def _prepare_enterprise_link(self, skills_fd: int) -> None:
        """Ensure ``skills/enterprise`` tracks ``managed`` before the commit.

        First apply (link absent): create the tracking symlink (dangling until
        the ``managed`` commit).  Updates (link present): verify it still has
        the expected target — a tampered link fails closed instead of being
        silently reclaimed.
        """
        try:
            st = _lstat_rel(_ENTERPRISE_LINK_NAME, skills_fd)
        except FileNotFoundError:
            temp_name = f".enterprise.swap.{secrets.token_hex(8)}"
            self._create_enterprise_symlink(skills_fd, temp_name)
            try:
                _atomic_replace(
                    temp_name, _ENTERPRISE_LINK_NAME,
                    src_dir_fd=skills_fd, dst_dir_fd=skills_fd,
                )
            except OSError:
                _unlink_rel(temp_name, skills_fd)
                raise
            self._fsync_enterprise_parent(skills_fd)
            return
        if not stat.S_ISLNK(st.st_mode):
            raise ProfileTamperError(
                "skills/enterprise must be a symlink, not a real directory"
            )
        target = os.readlink(_ENTERPRISE_LINK_NAME, dir_fd=skills_fd)
        if target != _ENTERPRISE_LINK_TARGET:
            raise ProfileTamperError("skills/enterprise symlink target is unexpected")

    def _create_enterprise_symlink(self, skills_fd: int, temp_name: str) -> None:
        try:
            _symlink_rel_fd(
                _ENTERPRISE_LINK_TARGET, temp_name, skills_fd,
                self._profile / "skills",
            )
        except FileExistsError:
            # Stale swap entry from a previous crashed apply; clear and retry once.
            _unlink_rel(temp_name, skills_fd)
            _symlink_rel_fd(
                _ENTERPRISE_LINK_TARGET, temp_name, skills_fd,
                self._profile / "skills",
            )

    def _fsync_enterprise_parent(self, skills_fd: int) -> None:
        _fsync_dir_fd(skills_fd)

    def _verify_entries_unchanged(
        self, profile_fd: int, versions_fd: int, skills_fd: int
    ) -> None:
        """Re-verify managed roots were not replaced (TOCTOU) right before commit."""
        self._verify_dir_entry_pinned(profile_fd, _VERSIONS_DIR_NAME, versions_fd)
        self._verify_dir_entry_pinned(profile_fd, "skills", skills_fd)
        # managed, if present, must still be a symlink (not swapped to a real dir).
        try:
            st = _lstat_rel(_MANAGED_LINK_NAME, profile_fd)
            if not stat.S_ISLNK(st.st_mode):
                raise ProfileTamperError("managed must be a symlink, not a real directory")
        except FileNotFoundError:
            pass  # first apply — managed does not exist yet
        # enterprise link, if present, must still track managed.
        try:
            est = _lstat_rel(_ENTERPRISE_LINK_NAME, skills_fd)
            if not stat.S_ISLNK(est.st_mode):
                raise ProfileTamperError(
                    "skills/enterprise must be a symlink, not a real directory"
                )
            if os.readlink(_ENTERPRISE_LINK_NAME, dir_fd=skills_fd) != _ENTERPRISE_LINK_TARGET:
                raise ProfileTamperError("skills/enterprise symlink target is unexpected")
        except FileNotFoundError:
            pass

    @staticmethod
    def _verify_dir_entry_pinned(parent_fd: int, name: str, held_fd: int) -> None:
        st_entry = _lstat_rel(name, parent_fd)
        if not stat.S_ISDIR(st_entry.st_mode):
            raise ProfileTamperError(f"{name} was replaced with a non-directory")
        st_held = os.fstat(held_fd)
        if st_entry.st_ino != st_held.st_ino or st_entry.st_dev != st_held.st_dev:
            raise ProfileTamperError(f"{name} directory was replaced mid-apply")

    def _commit_managed_link(self, profile_fd: int, version_key: str) -> None:
        """Atomically publish the ``managed`` symlink (the commit point).

        Creates a temp symlink and ``os.replace``s it over ``managed``.  This is
        the ONLY commit point: once ``os.replace`` returns, the new version is
        active.  No fallible operation (not even fsync) may run between the
        replace succeeding and the caller recording ``committed=True``; the
        post-commit durability fsync is performed separately by
        :meth:`_fsync_profile_dir` so its failure can never trigger a rollback
        that deletes the now-active version directory.
        """
        temp_name = f".managed.swap.{secrets.token_hex(8)}"
        target = f"{_VERSIONS_DIR_NAME}/{version_key}"
        try:
            _symlink_rel_fd(target, temp_name, profile_fd, self._profile)
        except OSError:
            _unlink_rel(temp_name, profile_fd)
            raise
        try:
            _atomic_replace(
                temp_name, _MANAGED_LINK_NAME,
                src_dir_fd=profile_fd, dst_dir_fd=profile_fd,
            )
        except OSError:
            _unlink_rel(temp_name, profile_fd)
            raise

    def _fsync_profile_dir(self, profile_fd: int) -> None:
        """Post-commit durability sync of the profile directory.

        Surfaces genuine fsync errors (e.g. EIO) so they can be reported as
        :class:`ProfileDurabilityUncertainError` without rolling back the
        committed state.  macOS ``EINVAL`` (fsync on a dir fd unsupported) is
        tolerated as before.
        """
        _fsync_dir_fd(profile_fd)

    # -- rollback --

    def _rollback(
        self,
        profile_fd: int,
        versions_fd: int,
        skills_fd: int,
        version_key: str,
        staging_name: Optional[str],
        first_apply: bool,
    ) -> None:
        if staging_name is not None:
            try:
                _remove_entry(versions_fd, staging_name)
            except FileNotFoundError:
                pass
        # Remove the newly-created version dir (orphan; not yet active).
        try:
            _lstat_rel(version_key, versions_fd)
            _remove_entry(versions_fd, version_key)
        except FileNotFoundError:
            pass
        # On first apply the enterprise link is dangling; remove it so no
        # half-baked active link survives.  On updates it tracks the still-active
        # old managed symlink and must be left in place.
        if first_apply:
            try:
                st = _lstat_rel(_ENTERPRISE_LINK_NAME, skills_fd)
                if stat.S_ISLNK(st.st_mode):
                    os.unlink(_ENTERPRISE_LINK_NAME, dir_fd=skills_fd)
            except FileNotFoundError:
                pass
        _fsync_dir_fd(versions_fd)
        _fsync_dir_fd(skills_fd)

    # -- cleanup --

    def _cleanup_old_versions(self, versions_fd: int, *, keep: str) -> None:
        for name in os.listdir(versions_fd):
            if name == keep:
                continue
            if name == _LOCK_FILE_NAME:
                continue
            if name.startswith(".staging.") or name.startswith(".managed.swap.") \
                    or name.startswith(".enterprise.swap."):
                _remove_entry(versions_fd, name)
                continue
            # A version directory.  lstat (no follow); skip symlinks, never
            # delete their target.
            try:
                st = _lstat_rel(name, versions_fd)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(st.st_mode):
                # Planted symlink inside .managed-versions: do not follow, do not
                # delete its target.  Leave it for forensic reporting.
                continue
            _remove_entry(versions_fd, name)
