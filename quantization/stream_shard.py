"""Bounded byte-preserving safetensors repacking for native checkpoint tensors.

No tensor decoding, dtype conversion, or whole-payload mapping is performed.
Destinations are published without overwriting existing files.
"""
import json
import os
from pathlib import Path
import struct
import tempfile


def read_header(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("truncated safetensors prefix")
        size, = struct.unpack("<Q", raw)
        if not 2 <= size <= 64 * 1024 * 1024:
            raise ValueError("invalid safetensors header size")
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated safetensors header")
        header = json.loads(data)
        payload_size = os.fstat(stream.fileno()).st_size - 8 - size
    spans = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        start, end = item["data_offsets"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= payload_size
                or not isinstance(item["dtype"], str) or not isinstance(item["shape"], list)
                or any(type(dim) is not int or dim < 0 for dim in item["shape"])):
            raise ValueError("invalid safetensors tensor descriptor")
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("safetensors payload has gaps or overlapping tensors")
        cursor = end
    if cursor != payload_size:
        raise ValueError("unindexed safetensors payload bytes")
    return header, 8 + size


def repack(entries, destination, *, chunk_bytes=8 * 1024 * 1024):
    """entries maps destination names to (source path, source tensor name)."""
    destination = Path(destination)
    if not entries or "__metadata__" in entries or not 1 <= chunk_bytes <= 64 * 1024 * 1024:
        raise ValueError("invalid repack inventory or buffer size")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("export destination already exists; verify it before explicit resume")
    headers, spans, header, cursor = {}, [], {}, 0
    for name, (path, source_name) in sorted(entries.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("destination tensor name is required")
        path = Path(path).resolve(strict=True)
        if path not in headers:
            headers[path] = read_header(path)
        source_header, base = headers[path]
        item = source_header[source_name]
        start, end = item["data_offsets"]
        header[name] = dict(dtype=item["dtype"], shape=item["shape"], data_offsets=[cursor, cursor + end - start])
        spans.append((path, base + start, end - start))
        cursor += end - start
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > 64 * 1024 * 1024:
        raise ValueError("export header exceeds bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".export-", suffix=".partial", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            prefix = struct.pack("<Q", len(encoded)) + encoded
            output.write(prefix)
            for path, offset, length in spans:
                with path.open("rb") as source:
                    source.seek(offset)
                    remaining = length
                    while remaining:
                        block = source.read(min(chunk_bytes, remaining))
                        if not block:
                            raise ValueError("source tensor payload truncated during export")
                        output.write(block)
                        remaining -= len(block)
            output.flush()
            os.fsync(output.fileno())
        # Atomic create-without-replace protects unrelated or completed output.
        os.link(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    expected_bytes = 8 + len(encoded) + cursor
    if destination.stat().st_size != expected_bytes:
        raise ValueError("exported shard size differs from planned tensor spans")
    return dict(bytes=expected_bytes, tensors=header)
