# SPDX-FileCopyrightText: 2026 Alex Gee
# SPDX-License-Identifier: GPL-3.0-or-later
"""State + staging + apply/rollback for opt-in GPU frame extraction.

Pure stdlib on purpose: this module is imported from ``__init__.py`` on
every plugin load (to apply a staged swap before anything imports cv2),
so it must never depend on cv2/torch/pycolmap. See the extractor probe
in ``core/sharpest_extractor.py`` for the runtime GPU capability check;
nothing here imports OpenCV.

The installed CUDA OpenCV build is version-camouflaged as the lock's CPU
pin so the host's ``uv sync`` leaves it in place. A build is identified
as CUDA by a ``CUDA_PATH`` default in its ``cv2/config.py`` (the CPU
wheel has none) — this is the single fingerprint used everywhere.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ".gpu_extraction_state.json"

_DEFAULT_STATE = {
    "state": "disabled",
    "detail": "",
    "staged_at": "",
    "wheel_sha256": "",
    "camouflage_version": "",
}


def get_state(root: Path = PLUGIN_ROOT) -> dict:
    """Return the state dict; the default (disabled) if no file exists.

    Reading never creates the file. A corrupt file degrades to default
    rather than raising — plugin load must never break on it.
    """
    path = Path(root) / STATE_FILE
    if not path.exists():
        return dict(_DEFAULT_STATE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)
    merged = dict(_DEFAULT_STATE)
    merged.update(data)
    return merged


def set_state(state: str, root: Path = PLUGIN_ROOT, **fields) -> None:
    """Write ``state`` plus any extra fields, preserving existing ones."""
    data = get_state(root=root)
    data["state"] = state
    data.update(fields)
    (Path(root) / STATE_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8")


def _site_packages(root: Path) -> Path:
    return Path(root) / ".venv" / "Lib" / "site-packages"


def _installed_build(root: Path) -> str:
    """Classify the installed cv2 as 'cuda', 'cpu', or 'missing'.

    'missing' covers both an absent ``cv2/config.py`` and an absent
    opencv dist-info — either is a crash remnant the host sync self-heals
    and, for an ``active`` state, means we were reverted.
    """
    sp = _site_packages(root)
    config = sp / "cv2" / "config.py"
    if not config.exists():
        return "missing"
    if not list(sp.glob("opencv_contrib_python-*.dist-info")):
        return "missing"
    text = config.read_text(encoding="utf-8", errors="replace")
    return "cuda" if "CUDA_PATH" in text else "cpu"


def detect_runtime_state(root: Path = PLUGIN_ROOT) -> str:
    """Reconcile the recorded state against the installed cv2.

    Only ``active`` is cross-checked: if the installed build is no longer
    the CUDA one (host sync or a manual action undid us, or a crash left
    no dist-info), transition to ``reverted`` and persist it. All other
    states pass through unchanged. Cheap and cv2-free; safe from the
    panel probe thread.
    """
    state = get_state(root=root)["state"]
    if state != "active":
        return state
    if _installed_build(root) == "cuda":
        return "active"
    set_state("reverted", root=root)
    return "reverted"


# --------------------------------------------------------------------------
# Staged swap / camouflage / rollback (apply_pending)
# --------------------------------------------------------------------------
#
# apply_pending is the ONLY function unsafe to call while cv2 is importable —
# it swaps cv2.pyd on disk. __init__.py calls it before any cv2 import.
#
# It is a crash-safe transaction: the process can die (power loss, kill) at
# ANY step. Two invariants make that safe:
#   * staging is COPIED, never moved, so a retry at next load can complete;
#   * the camouflaged (lock-satisfying) dist-info is written LAST, so no crash
#     window pairs it with an incomplete cv2/. Every mid-crash state either
#     has no opencv dist-info at all (the host's bare `uv sync` sees "missing"
#     and self-heals to the CPU baseline) or has a complete tree. The host
#     sync is the crash recovery, by construction.

_DISTINFO_PREFIX = "opencv_contrib_python-"
_DISTINFO_SUFFIX = ".dist-info"


def _staging(root: Path) -> Path:
    return Path(root) / "tmp" / "gpu-staging"


def _lib_gpu(root: Path) -> Path:
    return Path(root) / "lib" / "gpu"


def _rmtree_if(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def _distinfo_version(name: str) -> str:
    return name[len(_DISTINFO_PREFIX):-len(_DISTINFO_SUFFIX)]


def _find_distinfo(dirpath: Path) -> Path | None:
    hits = sorted(dirpath.glob(_DISTINFO_PREFIX + "*" + _DISTINFO_SUFFIX))
    return hits[0] if hits else None


def _camouflage_distinfo(sitepkgs: Path, from_ver: str, to_ver: str) -> None:
    """Rename the installed dist-info to ``to_ver`` and rewrite its
    self-references (METADATA Version, RECORD path prefixes). uv reconciles
    on recorded name+version+source, never on content, so the stale content
    hashes left in RECORD are ignored (proven 2026-07-10)."""
    dst = sitepkgs / (_DISTINFO_PREFIX + to_ver + _DISTINFO_SUFFIX)
    if from_ver != to_ver:
        (sitepkgs / (_DISTINFO_PREFIX + from_ver + _DISTINFO_SUFFIX)).rename(dst)
    meta = dst / "METADATA"
    meta.write_text(
        meta.read_text(encoding="utf-8").replace(
            "Version: " + from_ver, "Version: " + to_ver, 1),
        encoding="utf-8")
    rec = dst / "RECORD"
    rec.write_text(
        rec.read_text(encoding="utf-8").replace(
            _DISTINFO_PREFIX + from_ver + _DISTINFO_SUFFIX + "/",
            _DISTINFO_PREFIX + to_ver + _DISTINFO_SUFFIX + "/"),
        encoding="utf-8")


def _camouflage_target(root: Path) -> str:
    """Version to register the CUDA build as: whatever CPU pin is currently
    installed (dynamic, so a future lock bump keeps working). Falls back to
    the recorded field for a retry after the dist-info was already deleted."""
    di = _find_distinfo(_site_packages(root))
    if di is not None:
        return _distinfo_version(di.name)
    recorded = get_state(root=root).get("camouflage_version")
    if recorded:
        return recorded
    raise RuntimeError(
        "no installed opencv dist-info and no recorded camouflage_version")


def _enable_steps(root: Path, ctx: dict) -> list:
    sp = _site_packages(root)
    staging = _staging(root)
    staged_cv2 = staging / "wheel" / "cv2"
    staged_di = _find_distinfo(staging / "wheel")
    if staged_di is None:
        raise RuntimeError("no staged CUDA wheel dist-info")
    from_ver = _distinfo_version(staged_di.name)
    lib_src = staging / "lib-gpu"
    lib_dst = _lib_gpu(root)
    backup = staging / "backup"
    backup_tmp = staging / "backup.__tmp"
    incoming = sp / "cv2.__gpu_incoming"

    def prep():
        _rmtree_if(incoming)
        _rmtree_if(backup_tmp)
        ctx["to_ver"] = _camouflage_target(root)
        set_state("enable_staged", root=root, camouflage_version=ctx["to_ver"])

    def backup_copy():
        if backup.is_dir():
            return  # a completed backup is the only pristine tree; never touch
        backup_tmp.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sp / "cv2", backup_tmp / "cv2")
        di = _find_distinfo(sp)
        shutil.copytree(di, backup_tmp / di.name)

    def backup_commit():
        if backup.is_dir():
            return
        backup_tmp.rename(backup)

    def copy_incoming():
        shutil.copytree(staged_cv2, incoming)

    def delete_distinfo():
        di = _find_distinfo(sp)
        if di is not None:
            shutil.rmtree(di)

    def delete_cv2():
        _rmtree_if(sp / "cv2")

    def rename_incoming():
        incoming.rename(sp / "cv2")

    def copy_lib_gpu():
        lib_dst.parent.mkdir(parents=True, exist_ok=True)
        _rmtree_if(lib_dst)
        shutil.copytree(lib_src, lib_dst)

    def write_distinfo():
        dst = sp / staged_di.name
        _rmtree_if(dst)
        shutil.copytree(staged_di, dst)
        du = dst / "direct_url.json"
        if du.exists():
            du.unlink()  # uv reverts on recorded-source mismatch; must be absent
        _camouflage_distinfo(sp, from_ver, ctx["to_ver"])

    def finalize():
        set_state("active", root=root,
                  camouflage_version=ctx["to_ver"], build_version=from_ver)
        _rmtree_if(backup)
        _rmtree_if(backup_tmp)

    return [
        ("prep", prep),
        ("backup-copy", backup_copy),
        ("backup-commit", backup_commit),
        ("copy-incoming", copy_incoming),
        ("delete-distinfo", delete_distinfo),
        ("delete-cv2", delete_cv2),
        ("rename-incoming", rename_incoming),
        ("copy-lib-gpu", copy_lib_gpu),
        ("write-distinfo", write_distinfo),
        ("finalize", finalize),
    ]


def _disable_steps(root: Path, ctx: dict) -> list:
    sp = _site_packages(root)
    staging = _staging(root)
    staged_cv2 = staging / "wheel" / "cv2"
    staged_di = _find_distinfo(staging / "wheel")
    if staged_di is None:
        raise RuntimeError("no staged CPU wheel dist-info")
    lib_dst = _lib_gpu(root)
    backup = staging / "backup"
    backup_tmp = staging / "backup.__tmp"
    incoming = sp / "cv2.__gpu_incoming"

    def prep():
        _rmtree_if(incoming)
        _rmtree_if(backup_tmp)

    def backup_copy():
        if backup.is_dir():
            return
        backup_tmp.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sp / "cv2", backup_tmp / "cv2")
        di = _find_distinfo(sp)
        shutil.copytree(di, backup_tmp / di.name)

    def backup_commit():
        if backup.is_dir():
            return
        backup_tmp.rename(backup)

    def copy_incoming():
        shutil.copytree(staged_cv2, incoming)

    def delete_distinfo():
        di = _find_distinfo(sp)
        if di is not None:
            shutil.rmtree(di)

    def delete_cv2():
        _rmtree_if(sp / "cv2")

    def rename_incoming():
        incoming.rename(sp / "cv2")

    def write_distinfo():
        # CPU wheel keeps its REAL dist-info (no camouflage), just strip any
        # direct_url.json for symmetry with the enable path.
        dst = sp / staged_di.name
        _rmtree_if(dst)
        shutil.copytree(staged_di, dst)
        du = dst / "direct_url.json"
        if du.exists():
            du.unlink()

    def remove_lib_gpu():
        _rmtree_if(lib_dst)

    def finalize():
        set_state("disabled", root=root)
        _rmtree_if(backup)
        _rmtree_if(backup_tmp)

    return [
        ("prep", prep),
        ("backup-copy", backup_copy),
        ("backup-commit", backup_commit),
        ("copy-incoming", copy_incoming),
        ("delete-distinfo", delete_distinfo),
        ("delete-cv2", delete_cv2),
        ("rename-incoming", rename_incoming),
        ("write-distinfo", write_distinfo),
        ("remove-lib-gpu", remove_lib_gpu),
        ("finalize", finalize),
    ]


def _rollback(root: Path) -> None:
    """Restore ONLY from a completed backup/. If none exists, leave the tree
    as-is — a dist-info-less tree is exactly what the host sync self-heals.
    Never restore from backup.__tmp (it may be a partial copy)."""
    sp = _site_packages(root)
    _rmtree_if(sp / "cv2.__gpu_incoming")
    backup = _staging(root) / "backup"
    if not backup.is_dir():
        return
    _rmtree_if(sp / "cv2")
    for di in list(sp.glob(_DISTINFO_PREFIX + "*" + _DISTINFO_SUFFIX)):
        shutil.rmtree(di)
    for child in backup.iterdir():
        dst = sp / child.name
        _rmtree_if(dst)
        shutil.copytree(child, dst)


def apply_pending(root: Path = PLUGIN_ROOT, _after_step=None) -> str:
    """Apply a staged enable/disable. Called from __init__ before cv2 exists
    in-process. Acts only on ``*_staged`` states; any other state is a no-op.

    Never raises: any caught exception rolls back from a completed backup and
    sets state=error. ``_after_step(index, name)`` is a test-only seam — a
    SystemExit from it escapes uncaught (simulating process death, no
    rollback), an ordinary exception triggers the rollback path.
    """
    state = get_state(root=root)["state"]
    if state == "enable_staged":
        steps = _enable_steps(root, {})
    elif state == "disable_staged":
        steps = _disable_steps(root, {})
    else:
        return state
    try:
        for index, (name, fn) in enumerate(steps):
            fn()
            if _after_step is not None:
                _after_step(index, name)
    except Exception as exc:  # SystemExit is not Exception -> escapes uncaught
        _rollback(root)
        set_state("error", root=root,
                  detail="%s: %s" % (type(exc).__name__, exc))
        return "error"
    return get_state(root=root)["state"]
