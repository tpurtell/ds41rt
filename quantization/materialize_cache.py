"""Materialize acknowledged uploaded files into the standard HF cache.

The caller supplies a durable upload receipt binding remote blob IDs to local
file fingerprints. This is not an independent content verifier. No payload
hashing, copying, downloading, or overwriting existing cache blobs is performed.
"""
import fcntl
import os
from pathlib import Path
import re

from export_assets import asset_target, publish_asset
from write_export import _fingerprint, _sync_dir
import hashlib


def materialize_cache(output, cache_root, receipt):
    output, cache_root = Path(output).resolve(strict=True), Path(cache_root).resolve()
    repo = receipt["repo_id"]
    commit = receipt["commit"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or ".." in repo:
        raise ValueError("invalid cache repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or receipt.get("status") != "uploaded":
        raise ValueError("cache materialization requires an acknowledged commit")
    files = receipt["files"]
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() or path.is_symlink()}
    if not files or actual != set(files):
        raise ValueError("uploaded and local file inventories differ")
    # Validate all paths and acknowledged fingerprints before mutating the cache.
    blob_sources = {}
    for name, record in files.items():
        path = asset_target(output, name)
        if (not path.is_file() or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", record["blob"])
                or _fingerprint(path) != record["local"] or path.stat().st_size != record["bytes"]):
            raise ValueError("uploaded file identity differs from local export")
        blob_sources.setdefault(record["blob"], []).append(path)
    repo_root = asset_target(cache_root, "models--" + repo.replace("/", "--"))
    repo_root.mkdir(parents=True, exist_ok=True)
    lock_path = asset_target(repo_root, ".ds41rt-materialize.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blobs = asset_target(repo_root, "blobs")
        snapshot = asset_target(repo_root, "snapshots/" + commit)
        blobs.mkdir(exist_ok=True)
        snapshot.mkdir(parents=True, exist_ok=True)
        if output.stat().st_dev != blobs.stat().st_dev:
            raise ValueError("HF cache and export must share a filesystem for hard links")
        existing = {str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
                    if path.is_file() or path.is_symlink()}
        if existing - set(files):
            raise ValueError("cache snapshot contains unacknowledged files")
        for name, record in sorted(files.items()):
            source, blob = output / name, asset_target(blobs, record["blob"])
            if _fingerprint(source) != record["local"]:
                raise ValueError("export changed during cache materialization")
            if blob.exists():
                if not blob.is_file() or not any(os.path.samefile(candidate, blob)
                                                 for candidate in blob_sources[record["blob"]]):
                    raise ValueError("existing cache blob is not this acknowledged local file")
            else:
                os.link(source, blob)
                _sync_dir(blobs)
            # Parent directories must be real; the leaf is intentionally a symlink.
            parent_name = str(Path(name).parent)
            parent = snapshot if parent_name == "." else asset_target(snapshot, parent_name)
            parent.mkdir(parents=True, exist_ok=True)
            destination = parent / Path(name).name
            relative = os.path.relpath(blob, parent)
            if destination.is_symlink():
                if os.readlink(destination) != relative:
                    raise ValueError("existing cache snapshot link differs")
            elif destination.exists():
                raise ValueError("existing cache snapshot entry is not a symlink")
            else:
                destination.symlink_to(relative)
            for directory in (parent, *parent.parents):
                _sync_dir(directory)
                if directory == repo_root:
                    break
        payload = commit.encode()
        # Ref publication is last, so an interrupted run never advertises an
        # incomplete snapshot through main. A conflicting ref requires review.
        publish_asset(repo_root, "refs/main", dict(content=commit, bytes=len(payload),
                      sha256=hashlib.sha256(payload).hexdigest()))
        _sync_dir(cache_root)
    return dict(status="materialized", repo_id=repo, commit=commit,
                snapshot=str(snapshot), files=len(files), payload_copied_bytes=0)
