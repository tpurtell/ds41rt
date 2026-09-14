"""Resumable standard safetensors shard publication, without payload hashing.

This writes weights and the HF index, not model config/tokenizer or an upload.
State lives outside the artifact. Published shards are immutable; incomplete
temporary files are not accepted as shards and are never silently removed.
"""
import fcntl
import errno
import json
import os
from pathlib import Path
import tempfile

from export_layout import plan_shards
from stream_shard import read_header, repack


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json(path, value):
    if path.exists() or path.is_symlink():
        if path.is_symlink() or json.loads(path.read_text()) != value:
            raise ValueError(f"existing export metadata differs: {path.name}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_dir(path.parent)
    finally:
        os.unlink(temporary)


def _fingerprint(path):
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def verify_shard(path, names, entries):
    if path.is_symlink() or not path.is_file():
        raise ValueError("export shard must be a regular, non-symlink file")
    header, _ = read_header(path)
    if set(header) - {"__metadata__"} != set(names):
        raise ValueError("export shard tensor inventory differs")
    for name in names:
        actual, expected = header[name], entries[name]
        if (actual["dtype"] != expected["dtype"] or actual["shape"] != expected["shape"]
                or actual["data_offsets"][1] - actual["data_offsets"][0] != expected["bytes"]):
            raise ValueError("export shard tensor descriptor differs")


def _write_shard(destination, names, entries):
    paths = {Path(entries[name]["path"]).resolve(strict=True) for name in names}
    if len(paths) == 1 and all(entries[name]["tensor"] == name for name in names):
        source, = paths
        header, _ = read_header(source)
        if set(header) - {"__metadata__"} == set(names):
            verify_shard(source, names, entries)
            try:
                os.link(source, destination)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
            else:
                _sync_dir(destination.parent)
                return "hard-linked"
    repack({name: (entries[name]["path"], entries[name]["tensor"]) for name in names}, destination)
    return "written"


def write_export(inventory, output, state_root, *, identity, resume=False,
                 target_bytes=5_000_000_000, progress=None, metadata=None):
    """Write/explicitly resume a frozen inventory under an exclusive state lock.

    Resume checks source identity metadata and shard headers/sizes, not payload
    digests. Same-size payload corruption is outside this verification scope.
    Caller must provide the verified source/corpus/recipe/run identity.
    """
    if not isinstance(identity, dict) or not identity:
        raise ValueError("export identity is required")
    output, state_root = Path(output).resolve(), Path(state_root).resolve()
    if output == state_root or output.is_relative_to(state_root) or state_root.is_relative_to(output):
        raise ValueError("export state and artifact must be separate directory trees")
    progress = progress or (lambda event: None)
    metadata = metadata or {}
    if set(metadata) - {"config.json", "quantize_config.json"}:
        raise ValueError("unsupported export metadata filename")
    entries = inventory["tensors"]
    layout = plan_shards(entries, target_bytes=target_bytes)
    sources = {str(Path(item["path"]).resolve(strict=True)) for item in entries.values()}
    if any(Path(path).is_relative_to(output) for path in sources):
        raise ValueError("export cannot overwrite or contain its inputs")
    fingerprints = {path: _fingerprint(Path(path)) for path in sorted(sources)}
    plan = dict(version=1, identity=identity, output=str(output), inventory=inventory, metadata=metadata,
                layout=layout, sources=fingerprints)
    # JSON round-trip freezes types and rejects non-serializable state before IO.
    plan = json.loads(json.dumps(plan, allow_nan=False))
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "export.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = state_root / "plan.json"
        if plan_path.exists():
            if not resume:
                raise ValueError("existing export requires explicit resume")
            _publish_json(plan_path, plan)
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError("new export requires an empty artifact directory")
            _publish_json(plan_path, plan)
        output.mkdir(parents=True, exist_ok=True)
        unknown = {p.name for p in output.iterdir()} - set(layout["files"]) - {"model.safetensors.index.json"} - set(metadata)
        if unknown:
            raise ValueError("unexpected export files; inspect before recovery: " + ", ".join(sorted(unknown)))
        for filename, names in layout["files"].items():
            destination = output / filename
            if not destination.exists() and not destination.is_symlink():
                action = _write_shard(destination, names, entries)
            else:
                action = "reused"
            verify_shard(destination, names, entries)
            progress(dict(event="export_shard_verified", file=filename, action=action,
                          bytes=destination.stat().st_size))
        if any(_fingerprint(Path(path)) != expected for path, expected in fingerprints.items()):
            raise ValueError("source file metadata changed during export")
        _publish_json(output / "model.safetensors.index.json", layout["index"])
        for filename, value in metadata.items():
            _publish_json(output / filename, value)
        report = dict(status="weights-index-complete-model-validation-pending", files=len(layout["files"]),
                      tensors=len(entries), payload_bytes=layout["index"]["metadata"]["total_size"],
                      verification="source metadata and complete shard headers/sizes; no payload hashes")
        _publish_json(state_root / "weights-complete.json", report)
        return report
