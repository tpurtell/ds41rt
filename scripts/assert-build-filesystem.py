#!/usr/bin/env python3
"""Reject build paths on the unsafe scratch/NTFS filesystem, including bind mounts.

Usage: python3 scripts/assert-build-filesystem.py PATH [PATH ...]
Checks existing parents for paths that have not been created yet. No files are
created, and an unknown filesystem fails closed. Intended before mkdir/cargo.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


UNSAFE_TYPES = {"ntfs", "ntfs3", "fuseblk", "fuse.ntfs", "fuse.ntfs-3g"}


def check_path(value: str) -> None:
    path = Path(value).expanduser().resolve()
    if path == Path("/mnt/scratch") or Path("/mnt/scratch") in path.parents:
        raise ValueError(f"build path is on prohibited /mnt/scratch: {path}")
    existing = path
    while not existing.exists():
        if existing == existing.parent:
            raise ValueError(f"cannot locate filesystem for {path}")
        existing = existing.parent
    result = subprocess.run(
        ["findmnt", "--json", "--target", str(existing), "--output", "FSTYPE,OPTIONS"],
        check=True, capture_output=True, text=True,
    )
    mounts = json.loads(result.stdout).get("filesystems", [])
    if len(mounts) != 1 or not mounts[0].get("fstype"):
        raise ValueError(f"cannot determine filesystem for {path}")
    kind = mounts[0]["fstype"].lower()
    if kind in UNSAFE_TYPES or "ntfs" in kind:
        raise ValueError(f"build path {path} uses prohibited filesystem {kind} (possibly a bind mount)")
    if "ro" in mounts[0].get("options", "").split(","):
        raise ValueError(f"build path {path} is on a read-only filesystem")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        for value in argv:
            check_path(value)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"unsafe build filesystem: {error}\nUse ~/.cache/ds41rt/builds on root NVMe; never build on /mnt/scratch.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
