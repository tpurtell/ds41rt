"""Resumable preupload, one guarded commit, and durable local/remote receipt.

Call only after structural export validation. Failures propagate for explicit
recovery; this module adds no retry loop. SDK transport retries remain SDK-owned.
"""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils._runtime import is_xet_available

from export_assets import asset_target
from write_export import _fingerprint, _publish_json


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _remote(api, repo_id, commit):
    return {entry.path: dict(bytes=entry.size, blob=entry.lfs.sha256 if entry.lfs else entry.blob_id)
            for entry in api.list_repo_tree(repo_id, revision=commit, recursive=True, repo_type="model")
            if isinstance(entry, RepoFile)}


def _restored_operation(name, path, record):
    # Version-pinned SDK: construction reads only the first 512 bytes. Restoring
    # its documented mutated preupload state prevents retransmission on resume.
    operation = CommitOperationAdd(name, path)
    operation._upload_mode = record["mode"]
    operation._should_ignore = False
    if record["mode"] == "lfs":
        operation.upload_info.sha256 = bytes.fromhex(record["blob"])
        operation._is_uploaded = True
    return operation


def upload_artifact(output, state_root, repo_id, *, resume=False, api=None, progress=None):
    if importlib.metadata.version("huggingface_hub") != "1.26.1" or not is_xet_available():
        raise ValueError("upload requires qualified huggingface_hub 1.26.1 with Xet enabled")
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo_id)
    output, state_root = Path(output).resolve(strict=True), Path(state_root).resolve()
    if output == state_root or output.is_relative_to(state_root) or state_root.is_relative_to(output):
        raise ValueError("upload state must be separate from artifact")
    progress = progress or (lambda event: None)
    api = api or HfApi()
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() or path.is_symlink():
            name = str(path.relative_to(output))
            asset_target(output, name)
            files[name] = _fingerprint(path)
    if not {"README.md", ".gitattributes", "config.json", "model.safetensors.index.json"}.issubset(files):
        raise ValueError("upload requires complete model metadata and attributes")
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "upload.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = state_root / "upload-plan.json"
        if plan_path.exists() and not resume:
            raise ValueError("existing upload requires explicit resume")
        _publish_json(plan_path, dict(schema="ds41rt-upload-plan-v1", repo_id=repo_id,
                                     output=str(output), files=files))
        receipt_path = state_root / "upload-complete.json"
        receipt = _read(receipt_path)
        if receipt is not None:
            expected = {name: {key: record[key] for key in ("bytes", "blob")}
                        for name, record in receipt["files"].items()}
            if (receipt["repo_id"] != repo_id or set(receipt["files"]) != set(files)
                    or any(receipt["files"][name]["local"] != local for name, local in files.items())
                    or _remote(api, repo_id, receipt["commit"]) != expected):
                raise ValueError("completed upload receipt differs")
            return receipt
        parent_path = state_root / "upload-parent.json"
        parent = _read(parent_path)
        if parent is None:
            try:
                info = api.repo_info(repo_id, repo_type="model")
            except RepositoryNotFoundError:
                api.create_repo(repo_id, repo_type="model", private=False, exist_ok=False)
                info = api.repo_info(repo_id, repo_type="model")
            if set(_remote(api, repo_id, info.sha)) - {".gitattributes"}:
                raise ValueError("refusing to overwrite an existing populated repository")
            parent = dict(commit=info.sha)
            _publish_json(parent_path, parent)
        prepared, operations = {}, []
        records_root = state_root / "preuploaded"
        records_root.mkdir(exist_ok=True)
        for ordinal, (name, local) in enumerate(files.items()):
            path = output / name
            if _fingerprint(path) != local:
                raise ValueError("export changed during upload")
            record_path = records_root / f"{ordinal:06d}.json"
            record = _read(record_path)
            if record is None:
                progress(dict(event="upload_file_started", file=name, bytes=local["bytes"]))
                operation = CommitOperationAdd(name, path)
                api.preupload_lfs_files(repo_id, [operation], repo_type="model", revision="main",
                                        num_threads=1, free_memory=False, gitignore_content="")
                if operation._should_ignore or operation._upload_mode not in {"regular", "lfs"}:
                    raise ValueError("server ignored or did not classify an export file")
                if operation._upload_mode == "lfs":
                    if not operation._is_uploaded or not operation.upload_info.is_hashed:
                        raise ValueError("Xet did not return the completed upload hash")
                    blob = operation.upload_info.sha256.hex()
                else:
                    if local["bytes"] > 64 * 1024 * 1024:
                        raise ValueError("unexpected oversized regular Git upload")
                    payload = path.read_bytes()
                    blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                if _fingerprint(path) != local:
                    raise ValueError("export changed during preupload")
                record = dict(name=name, local=local, bytes=local["bytes"], blob=blob,
                              mode=operation._upload_mode)
                _publish_json(record_path, record)
                progress(dict(event="upload_file_prepared", file=name, bytes=local["bytes"], mode=record["mode"]))
            else:
                if record["name"] != name or record["local"] != local:
                    raise ValueError("preupload receipt differs from export")
                operation = _restored_operation(name, path, record)
                progress(dict(event="upload_file_reused", file=name, bytes=local["bytes"]))
            prepared[name] = dict(local=local, bytes=record["bytes"], blob=record["blob"])
            operations.append(operation)
        expected = {name: {key: record[key] for key in ("bytes", "blob")}
                    for name, record in prepared.items()}
        head = api.repo_info(repo_id, repo_type="model").sha
        if head != parent["commit"]:
            # Covers lost commit responses. Do not overwrite an unrelated writer.
            if _remote(api, repo_id, head) != expected:
                raise ValueError("remote repository changed during upload")
            commit = head
        else:
            if any(_fingerprint(output / name) != local for name, local in files.items()):
                raise ValueError("export changed before commit")
            commit = api.create_commit(repo_id, operations, repo_type="model", revision="main",
                parent_commit=parent["commit"], commit_message="Publish routed-only V4.1 EXL3 K3.25 model").oid
        if _remote(api, repo_id, commit) != expected:
            raise ValueError("committed remote inventory differs from uploaded files")
        if any(_fingerprint(output / name) != local for name, local in files.items()):
            raise ValueError("export changed before upload receipt")
        receipt = dict(schema="ds41rt-upload-receipt-v1", status="uploaded", repo_id=repo_id,
                       commit=commit, files=prepared)
        _publish_json(receipt_path, receipt)
        progress(dict(event="model_uploaded", repo_id=repo_id, commit=commit, files=len(prepared)))
        return receipt
