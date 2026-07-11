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

import ctypes
import datetime
import hashlib
import io
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
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


# --------------------------------------------------------------------------
# Staging (download + verify + unpack) — stage_enable
# --------------------------------------------------------------------------
#
# Runs from the panel's install thread while LFS is up; writes ONLY to
# tmp/gpu-staging/ (never .venv). apply_pending later consumes the staged
# layout at the next load. Sizes/hashes are pinned from
# docs/gpu-dll-manifest.md (Task 0.2, the source of truth). The CUDA overlay
# is OpenCV 4.12.0.88 (CUDA 12); it is camouflaged to the lock's CPU pin at
# apply time, not here.

_CUDA_WHEEL = {
    "url": "https://github.com/cudawarped/opencv-python-cuda-wheels/releases/"
           "download/4.12.0.88/"
           "opencv_contrib_python-4.12.0.88-cp37-abi3-win_amd64.whl",
    "sha256": "061b6c3594ddee34573f4e0e3257f9c1f5069985488f869f46305918cc9a8f52",
    "version": "4.12.0.88",
}

# Whole-wheel downloads (we need ~all of each): url, sha256, needed DLLs.
_NVIDIA_WHEELS = [
    {"name": "nvidia-npp-cu12",
     "url": "https://files.pythonhosted.org/packages/ae/91/"
            "e5f3067f369ce9ff3b35613a3e14bb230a17d4d1fb62390087ef90d9c235/"
            "nvidia_npp_cu12-12.4.1.87-py3-none-win_amd64.whl",
     "sha256": "7c425c400b610eecfb1a08cfc92ecfa4a1927c2ecb691bc26406444c605d30a9",
     "dlls": ["nppc64_12.dll", "nppial64_12.dll", "nppicc64_12.dll",
              "nppidei64_12.dll", "nppif64_12.dll", "nppig64_12.dll",
              "nppim64_12.dll", "nppist64_12.dll", "nppitc64_12.dll"]},
    {"name": "nvidia-cublas-cu12",
     "url": "https://files.pythonhosted.org/packages/20/e2/"
            "fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
            "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
     "sha256": "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661",
     "dlls": ["cublas64_12.dll", "cublasLt64_12.dll"]},
    {"name": "nvidia-cufft-cu12",
     "url": "https://files.pythonhosted.org/packages/20/ee/"
            "29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/"
            "nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
     "sha256": "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
     "dlls": ["cufft64_11.dll"]},
]

# cuDNN: range-extract ONLY the 0.3 MB dispatcher (its 1070 MB of engine DLLs
# are runtime-LoadLibrary'd, not needed to import cv2 — see manifest). Guarded
# by a per-file sha256; whole-wheel download is the fallback if Range fails.
_CUDNN_WHEEL = {
    "name": "nvidia-cudnn-cu12",
    "url": "https://files.pythonhosted.org/packages/29/28/"
           "2c9a2a97a8b3fedcf74a14f38fd5edfae12274380a829fdc6b16ce29be4c/"
           "nvidia_cudnn_cu12-9.24.0.43-py3-none-win_amd64.whl",
    "dll": "cudnn64_9.dll",
    "dll_sha256": "cccabec1388fd7f93d7067536a3489c8e7ec31405c2984c1f78c0e3d36600fbf",
}


def _default_sources() -> dict:
    return {"cuda_wheel": _CUDA_WHEEL,
            "nvidia_wheels": _NVIDIA_WHEELS,
            "cudnn": _CUDNN_WHEEL}


def _emit(on_output, msg: str) -> None:
    if on_output is not None:
        on_output(msg)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _reset_dir(path: Path) -> None:
    _rmtree_if(path)
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HttpRangeFile(io.RawIOBase):
    """Seekable file-like backed by HTTP Range requests, so stdlib zipfile can
    read a single wheel entry without downloading the whole wheel."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url, headers={"Range": "bytes=%d-%d" % (self.pos, end)})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        self.pos += len(data)
        return data


def _download(url: str, dest: Path, on_output=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _emit(on_output, "Downloading %s ..." % dest.name)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _fetch_verified(url: str, sha256: str, dest: Path, on_output=None) -> Path:
    if dest.exists() and _sha256_file(dest) == sha256:
        _emit(on_output, "Using cached %s" % dest.name)
        return dest
    _download(url, dest, on_output)
    got = _sha256_file(dest)
    if got != sha256:
        try:
            dest.unlink()
        except OSError:
            pass
        raise ValueError("hash mismatch for %s (%s != %s)" % (dest.name, got, sha256))
    return dest


def _url_to_zip(url: str) -> zipfile.ZipFile:
    """Open a wheel as a seekable zip: HTTP uses Range; file:// / paths open
    the local file directly."""
    if url.startswith(("http://", "https://")):
        return zipfile.ZipFile(HttpRangeFile(url))
    if url.startswith("file://"):
        url = urllib.request.url2pathname(urllib.parse.urlparse(url).path)
    return zipfile.ZipFile(url)


def _find_entry(zf: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    for info in zf.infolist():
        if info.filename.rsplit("/", 1)[-1] == basename:
            return info
    raise KeyError("%s not found in wheel" % basename)


def _selective_read(url: str, basename: str) -> bytes:
    with _url_to_zip(url) as zf:
        return zf.read(_find_entry(zf, basename))


def _verify_and_write(data: bytes, sha256: str, dest: Path) -> None:
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise ValueError("hash mismatch for %s (%s != %s)" % (dest.name, got, sha256))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)


def _extract_dlls(wheel: Path, dll_names, out_dir: Path, on_output=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(dll_names)
    with zipfile.ZipFile(wheel) as zf:
        for info in zf.infolist():
            base = info.filename.rsplit("/", 1)[-1]
            if base in wanted:
                (out_dir / base).write_bytes(zf.read(info))
                _emit(on_output, "  extracted %s" % base)


def _extract_cudnn(spec: dict, lib_gpu: Path, cache: Path, on_output=None) -> None:
    dest = lib_gpu / spec["dll"]
    try:
        _emit(on_output, "Fetching cuDNN dispatcher...")
        data = _selective_read(spec["url"], spec["dll"])
    except Exception as exc:  # Range unsupported / network -> whole-wheel fallback
        _emit(on_output,
              "Range fetch failed (%s); downloading full cuDNN wheel" % exc)
        whl = cache / (spec["name"] + ".whl")
        _download(spec["url"], whl, on_output)
        with zipfile.ZipFile(whl) as zf:
            data = zf.read(_find_entry(zf, spec["dll"]))
    _verify_and_write(data, spec["dll_sha256"], dest)


def _unpack_wheel(wheel: Path, out: Path, on_output=None) -> None:
    _reset_dir(out)
    _emit(on_output, "Unpacking OpenCV build...")
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(out)


def stage_enable(on_output=None, root: Path = PLUGIN_ROOT, _sources=None) -> bool:
    """Download + verify + unpack the CUDA overlay into tmp/gpu-staging/.
    Returns True on success (state -> enable_staged), False on any failure
    (state unchanged, partial staging cleaned). Safe while LFS runs.

    ``_sources`` is a test seam (file:// mini-wheels); production uses the
    pinned constants above.
    """
    src = _sources if _sources is not None else _default_sources()
    staging = _staging(root)
    cache = staging / "cache"
    try:
        cuda = src["cuda_wheel"]
        whl = _fetch_verified(cuda["url"], cuda["sha256"],
                              cache / "opencv-cuda.whl", on_output)
        _unpack_wheel(whl, staging / "wheel", on_output)

        lib_gpu = staging / "lib-gpu"
        _reset_dir(lib_gpu)
        for spec in src["nvidia_wheels"]:
            w = _fetch_verified(spec["url"], spec["sha256"],
                                cache / (spec["name"] + ".whl"), on_output)
            _extract_dlls(w, spec["dlls"], lib_gpu, on_output)
        _extract_cudnn(src["cudnn"], lib_gpu, cache, on_output)

        try:
            cam = _camouflage_target(root)
        except RuntimeError:
            cam = ""  # apply_pending recomputes from the installed dist-info
        set_state("enable_staged", root=root,
                  wheel_sha256=cuda["sha256"], camouflage_version=cam,
                  staged_at=_now())
        _emit(on_output,
              "GPU extraction staged. Restart LichtFeld Studio to activate.")
        return True
    except Exception as exc:
        _emit(on_output, "GPU setup failed: %s" % exc)
        _rmtree_if(staging / "wheel")
        _rmtree_if(staging / "lib-gpu")
        return False


# --------------------------------------------------------------------------
# Panel backend helpers
# --------------------------------------------------------------------------

def cancel_staged(root: Path = PLUGIN_ROOT) -> str:
    """Cancel a staged enable: delete the staged wheel + lib-gpu and return to
    disabled. Keeps the download cache so a re-enable is cheap. No-op for
    non-staged states."""
    state = get_state(root=root)["state"]
    staging = _staging(root)
    if state == "enable_staged":
        _rmtree_if(staging / "wheel")
        _rmtree_if(staging / "lib-gpu")
        set_state("disabled", root=root)
        return "disabled"
    return state


def gpu_hardware_present() -> bool:
    """Light probe, no cv2/torch: does the NVIDIA driver's nvcuda.dll load?
    Proves an NVIDIA GPU/driver exists — NOT that the CUDA build will run."""
    if sys.platform != "win32":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


def describe_installed_build(root: Path = PLUGIN_ROOT) -> str:
    """Truthful diagnostics (counterweight to the version disguise): package
    tools inside the venv report the registered version, so bug reports need
    this counter-truth. Real build version comes from the state file marker
    written at apply time, cross-checked against the config.py fingerprint."""
    build = _installed_build(root)
    if build == "missing":
        return "No OpenCV build installed"
    di = _find_distinfo(_site_packages(root))
    registered = _distinfo_version(di.name) if di is not None else "unknown"
    if build == "cuda":
        real = get_state(root=root).get("build_version") or "unknown"
        return "CUDA build %s (registered as %s)" % (real, registered)
    return "CPU build %s" % registered


def diagnostics_text(root: Path = PLUGIN_ROOT) -> str:
    """A paste-ready diagnostic bundle for bug reports. The 'installed build'
    line is the point: it reports the TRUE build, which venv package tools
    misreport because of the version disguise."""
    st = get_state(root=root)
    return "\n".join([
        "360 Plugin - GPU extraction diagnostics",
        "state: %s" % st["state"],
        "detail: %s" % (st.get("detail") or "(none)"),
        "installed build: %s" % describe_installed_build(root=root),
        "nvidia gpu detected: %s" % ("yes" if gpu_hardware_present() else "no"),
    ])
