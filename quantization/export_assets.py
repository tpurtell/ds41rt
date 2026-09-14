"""Preserve bounded checkpoint assets byte-for-byte, separate from weight IO."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile


def asset_target(root, name):
    relative = PurePosixPath(name)
    if not name or str(relative) != name or relative.is_absolute() or any(part in (".", "..") for part in name.split("/")) or "\\" in name:
        raise ValueError("unsafe export asset path")
    target = root.joinpath(*relative.parts)
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != root.parent):
        raise ValueError("export assets cannot traverse symlinks")
    return target


def plan_assets(snapshot, attestation):
    if attestation.get("status") != "passed" or attestation.get("source_snapshot") != snapshot.name:
        raise ValueError("assets require the passed source attestation")
    assets = {}
    for name, record in sorted(attestation["files"].items()):
        if name.endswith(".safetensors") or name in {"config.json", "model.safetensors.index.json"}:
            continue
        destination = "README.source.md" if name == "README.md" else name
        # The source card describes the unquantized checkpoint and must not
        # become this quantization's HF model card by accident.
        if destination in assets or not 0 <= record["bytes"] <= 64 * 1024 * 1024:
            raise ValueError("duplicate or oversized checkpoint asset")
        asset_target(Path("/export"), destination)
        source = snapshot / name
        if not source.is_file() or source.stat().st_size != record["bytes"]:
            raise ValueError("source asset differs from attestation")
        assets[destination] = dict(path=str(source), bytes=record["bytes"], sha256=record["sha256"])
    if not {"tokenizer.json", "tokenizer_config.json", "LICENSE", "inference/kernel.py"}.issubset(assets):
        raise ValueError("checkpoint asset inventory is incomplete")
    return assets


def publish_asset(root, name, record):
    target = asset_target(root, name)
    if not 0 <= record["bytes"] <= 64 * 1024 * 1024:
        raise ValueError("asset exceeds bounded small-file policy")
    if target.exists():
        if not target.is_file() or target.stat().st_size != record["bytes"]:
            raise ValueError("existing export asset size differs")
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError("existing export asset checksum differs")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".asset-", suffix=".partial", dir=target.parent)
    try:
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as output, Path(record["path"]).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > record["bytes"]:
                    raise ValueError("source asset grew during export")
                digest.update(chunk)
                output.write(chunk)
            if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError("source asset checksum differs")
            output.flush()
            os.fchmod(output.fileno(), 0o644)
            os.fsync(output.fileno())
        os.link(temporary, target)
        # Sync all created directory entries through the artifact root.
        for directory in (target.parent, *target.parent.parents):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if directory == root:
                break
    finally:
        os.unlink(temporary)
