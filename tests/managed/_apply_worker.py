"""Subprocess worker for cross-process concurrency tests.

Run as a separate Python process (not forked from the test process) so the
``fcntl`` apply lock is exercised between truly independent processes — this
avoids ``fork()``-from-a-multi-threaded-pytest-worker deadlocks and is a
stronger test of the cross-process guarantee than threads or fork.

Usage::

    python _apply_worker.py apply <profile> <revision> <version_id> <skills_json> [pause]
    python _apply_worker.py hold <profile> <seconds>

Output: a single JSON line on stdout.
"""

import json
import sys
import time
from pathlib import Path


def _build_assignment(revision, version_id, skills):
    from hermes_managed.contracts import canonical_json_bytes, parse_runtime_assignment, sha256_hex

    policy = {
        "mode": "ENTERPRISE_MANAGED",
        "allowedModels": ["enterprise/deepseek-chat"],
        "defaultModel": "enterprise/deepseek-chat",
        "fallbackModels": [],
        "localProviderAllowed": False,
        "policyVersion": "sha256:5ea1e6720cc6",
        "policySha256": "5ea1e6720cc6dc3ef2cae5579c687dd98d80245bdf3a2ca89fb70e85c876899a",
    }
    manifest = {
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
        "enterpriseSkills": skills,
    }
    payload = {
        "assignmentId": 20,
        "revision": int(revision),
        "templateId": 10,
        "versionId": int(version_id),
        "templateSlug": "reference-assistant",
        "displayName": "Reference Assistant",
        "manifest": manifest,
        "manifestSha256": sha256_hex(canonical_json_bytes(manifest)),
        "effectiveModelPolicy": policy,
        "revoked": False,
        "revokedAt": None,
    }
    return parse_runtime_assignment(payload)


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _apply(profile, revision, version_id, skills, pause):
    from hermes_managed.profile_applier import ProfileApplier

    applier = ProfileApplier(Path(profile))
    if pause:
        applier._pre_commit_hook = lambda: time.sleep(pause)
    assignment = _build_assignment(revision, version_id, skills)
    result = applier.apply(assignment, lock_timeout=30.0)
    _emit({"ok": True, "status": result.status, "revision": result.revision})


def _hold(profile, seconds):
    import fcntl
    import os

    p = Path(profile)
    versions = p / ".managed-versions"
    versions.mkdir(parents=True, exist_ok=True)
    lock_path = versions / ".apply.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _emit({"ok": True, "holding": True})
        time.sleep(float(seconds))
    finally:
        os.close(fd)


def main():
    mode = sys.argv[1]
    if mode == "apply":
        _, _, profile, revision, version_id, skills_json, *rest = sys.argv
        pause = float(rest[0]) if rest else 0.0
        skills = json.loads(skills_json)
        try:
            _apply(profile, revision, version_id, skills, pause)
        except Exception as exc:  # noqa: BLE001
            _emit({"ok": False, "error": type(exc).__name__, "msg": str(exc)})
    elif mode == "hold":
        _, _, profile, seconds = sys.argv
        _hold(profile, seconds)
    else:
        _emit({"ok": False, "error": "unknown mode", "msg": mode})


if __name__ == "__main__":
    main()
