"""Durable immutable artifact journal for explicit quantization recovery.

Files are published first. A journal record is committed only after its payload
and every named dependency record exist. Orphan files are not completed work. No retry,
cleanup, process restart or code-upgrade authorization is inferred here.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RunStore:
    def __init__(self, root, identity):
        if not isinstance(identity, dict) or not identity:
            raise ValueError("run identity is required")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "run.sqlite", timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS artifacts (key TEXT PRIMARY KEY, record TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS retirements (key TEXT PRIMARY KEY, barrier TEXT NOT NULL, intent TEXT NOT NULL)")
        try:
            with self.db:
                self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (canonical(identity),))
                actual = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]
                if actual != canonical(identity):
                    raise ValueError("run identity mismatch; explicit recovery migration is required")
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def _path(self, path):
        path = Path(path)
        path = (self.root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file() or path.name.startswith("run.sqlite"):
            raise ValueError("artifact must be a payload file inside the run root")
        return path

    @staticmethod
    def _hash(path, sync=False):
        with path.open("rb") as stream:
            result = hashlib.file_digest(stream, "sha256").hexdigest()
            if sync:
                os.fsync(stream.fileno())
            return result

    def record_file(self, key, kind, path, *, parents=()):
        if not isinstance(key, str) or not key or not isinstance(kind, str) or not kind:
            raise ValueError("artifact key and kind are required")
        if len(set(parents)) != len(parents) or key in parents:
            raise ValueError("invalid artifact dependencies")
        path = self._path(path)
        checksum = self._hash(path, sync=True)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            dependencies = {}
            for parent in parents:
                row = self.db.execute("SELECT record FROM artifacts WHERE key=?", (parent,)).fetchone()
                if row is None:
                    raise ValueError(f"uncommitted dependency: {parent}")
                dependencies[parent] = hashlib.sha256(row[0].encode()).hexdigest()
            record = dict(kind=kind, path=str(path.relative_to(self.root)), sha256=checksum,
                          bytes=path.stat().st_size, parents=dependencies)
            encoded = canonical(record)
            existing = self.db.execute("SELECT record FROM artifacts WHERE key=?", (key,)).fetchone()
            if existing is not None and existing[0] != encoded:
                raise ValueError("cannot replace an already-committed artifact")
            self.db.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?)", (key, encoded))
        return record

    def get(self, key, *, verify=True):
        row = self.db.execute("SELECT record FROM artifacts WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        record = json.loads(row[0])
        if verify:
            if self.db.execute("SELECT 1 FROM retirements WHERE key=?", (key,)).fetchone():
                raise ValueError(f"artifact payload has been retired: {key}")
            for parent, checksum in record["parents"].items():
                parent_row = self.db.execute("SELECT record FROM artifacts WHERE key=?", (parent,)).fetchone()
                if parent_row is None or hashlib.sha256(parent_row[0].encode()).hexdigest() != checksum:
                    raise ValueError(f"committed dependency record is corrupt: {parent}")
            path = self._path(record["path"])
            if path.stat().st_size != record["bytes"] or self._hash(path) != record["sha256"]:
                raise ValueError(f"committed artifact is corrupt: {key}")
        return record

    def retire_files(self, keys, *, barrier):
        """Retire explicit temporary ancestors of a durable completed block.

        The caller must validate that the barrier's outputs and selected weights
        are usable and exclude anything still needed. Immutable artifact records
        survive retirement. Durable intents precede unlink, making interrupted
        retirement retryable without confusing missing files with corruption.
        """
        keys = tuple(keys)
        if not keys or len(set(keys)) != len(keys) or barrier in keys:
            raise ValueError("retirement requires distinct explicit payload keys")
        barrier_record = self.get(barrier)
        if barrier_record is None or barrier_record["kind"] not in {"block", "namespace-retirement"}:
            raise ValueError("retirement requires a completed block barrier")
        ancestors, pending = set(), list(barrier_record["parents"].items())
        while pending:
            key, expected_hash = pending.pop()
            record = self.get(key, verify=False)
            if record is None or hashlib.sha256(canonical(record).encode()).hexdigest() != expected_hash:
                raise ValueError("barrier has a corrupt dependency record")
            if key in ancestors:
                continue
            ancestors.add(key)
            pending.extend(record["parents"].items())
        path_keys = {}
        for other, encoded in self.db.execute("SELECT key, record FROM artifacts"):
            path_keys.setdefault(json.loads(encoded)["path"], []).append(other)
        intents = []
        for key in keys:
            record = self.get(key, verify=False)
            if key not in ancestors or record["kind"] not in {"hessian", "routed", "replay", "projection"}:
                raise ValueError("payload is not a temporary ancestor of the barrier")
            intent = canonical(dict(record=record, barrier_record=barrier_record))
            old = self.db.execute("SELECT barrier, intent FROM retirements WHERE key=?", (key,)).fetchone()
            if old is not None and old != (barrier, intent):
                raise ValueError("retirement identity changed")
            path = self.root / record["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError("retirement target must be a regular internal payload")
            # One payload path must not simultaneously represent a retained key.
            if path_keys[record["path"]] != [key]:
                raise ValueError("retirement payload has an aliased artifact key")
            if path.exists():
                if not path.is_file() or self._hash(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                    raise ValueError("retirement target is corrupt")
            elif old is None:
                raise ValueError("retirement target was missing before intent")
            intents.append((key, intent, path))
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO retirements VALUES (?, ?, ?)",
                                [(key, barrier, intent) for key, intent, _ in intents])
        removed = 0
        for key, intent, path in intents:
            if path.exists():
                removed += path.stat().st_size
                path.unlink()
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return removed
