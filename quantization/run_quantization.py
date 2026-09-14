#!/usr/bin/env python3
"""Manifest-driven two-RTX/four-Spark quantization and weight export stage.

Not yet the complete deployment/validation/upload launcher. Failures propagate;
there is no automatic restart or retry. Logs survive the invoking chat session.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys

import torch
from tokenicer import Tokenicer
from transformers import DeepseekV41Config
from transformers.models.deepseek_v41.modeling_deepseek_v41 import DeepseekV41NgramHashState

from gptqmodel.utils.v41_source import V41Source
from gptqmodel.utils.exl3_remote import EXL3RemoteClient, RemoteEndpoint, CoordinatorSlot
from block_driver import BlockDriver
from coordinator import quantize_namespaces
from corpus_inputs import tokenize_corpus
from distributed_search import DistributedSearch
from export_inventory import build_inventory
from export_config import model_metadata
from export_assets import plan_assets
from model_card import model_card_asset
from run_store import RunStore
from write_export import write_export
from validate_export import validate_export


def verify_source(snapshot, report_path):
    """Reuse the paid source attestation; never hash large checkpoint shards."""
    report = json.loads(Path(report_path).read_text())
    core = {key: value for key, value in report.items() if key not in {"manifest_sha256", "status"}}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (report.get("schema") != "ds41rt-v41-source-attestation-v1" or report.get("status") != "passed"
            or report.get("manifest_sha256") != digest or report.get("source_snapshot") != snapshot.name):
        raise ValueError("source attestation identity mismatch")
    paths = {str(path.relative_to(snapshot)): path for path in snapshot.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if set(paths) != set(report["files"]):
        raise ValueError("source file inventory differs from attestation")
    for name, path in paths.items():
        record = report["files"][name]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"source file size differs: {name}")
        if path.suffix == ".safetensors":
            # A content-addressed HF symlink can be checked without reading its
            # payload. Regular files get size/header checks, not a hash claim.
            if path.is_symlink() and path.resolve().name != record["sha256"]:
                raise ValueError(f"source blob address differs: {name}")
        else:
            if record["bytes"] > 64 * 1024 * 1024:
                raise ValueError("unexpected large non-weight source asset")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                    raise ValueError(f"source asset differs: {name}")
    from export_inventory import source_entries
    entries = source_entries(snapshot)
    if len(entries) != report["tensor_count"]:
        raise ValueError("source tensor count differs from attestation")
    return digest


def read_attestation(path):
    """Accept a standalone report or the original append-only diagnostic log."""
    payload = Path(path).read_text()
    try:
        standalone = json.loads(payload)
    except json.JSONDecodeError:
        standalone = None
    if isinstance(standalone, dict):
        if standalone.get("schema") != "ds41rt-input-attestation-v1" or standalone.get("status") != "passed":
            raise ValueError("expected a passed input attestation")
        return standalone
    reports = []
    for line in payload.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "ds41rt-input-attestation-v1":
            reports.append(value)
    if len(reports) != 1 or reports[0].get("status") != "passed":
        raise ValueError("expected exactly one passed input attestation")
    return reports[0]


def prepare_inputs(snapshot, corpus, attestation):
    """Check small attested assets and tokenize unchanged corpus before GPU IO."""
    for filename, expected in attestation["source_files"].items():
        path = snapshot / filename
        if not path.resolve().is_file() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("input attestation assets must be bounded small files")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"input asset differs from attestation: {filename}")
    config = DeepseekV41Config.from_pretrained(snapshot, local_files_only=True)
    tokenizer = Tokenicer.load(str(snapshot), model_config=config, local_files_only=True,
                               trust_remote_code=False).tokenizer
    records = tokenize_corpus(corpus, tokenizer, attestation)
    hashes = DeepseekV41NgramHashState(config.get_text_config())
    hashes.bind_tokenizer(tokenizer)
    return records, hashes


def validate_manifest(manifest):
    if manifest.get("schema") != "ds41rt-quantization-runtime-v1":
        raise ValueError("unsupported runtime manifest")
    required = {"identity", "snapshot", "source_attestation", "corpus", "input_attestation", "run_root", "output",
                "export_state", "token_file", "endpoints", "coordinator_slots"}
    if required - set(manifest) or not isinstance(manifest["identity"], dict) or not manifest["identity"]:
        raise ValueError("incomplete runtime manifest")
    if ({item["name"] for item in manifest["endpoints"]} != {"ostrich", "dodo", "emu", "kiwi"}
            or len(manifest["endpoints"]) != 4
            or [item["device"] for item in manifest["coordinator_slots"]] != ["cuda:0", "cuda:1"]):
        raise ValueError("runtime requires both RTX GPUs and all four Sparks")
    for item in manifest["endpoints"] + manifest["coordinator_slots"]:
        if not item.get("preflight_sha256") or not item.get("image_digest"):
            raise ValueError("runtime slots require pinned qualification identities")
    for name in ("snapshot", "source_attestation", "corpus", "input_attestation", "run_root", "output", "export_state", "token_file"):
        if not Path(manifest[name]).is_absolute():
            raise ValueError("runtime paths must be absolute")


def run(manifest, *, resume=False):
    validate_manifest(manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(manifest["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "runtime.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "run.sqlite").exists() and not resume:
            raise ValueError("existing quantization journal requires explicit --resume")
        with (root / "runtime-events.jsonl").open("a", buffering=1) as log:
            def progress(event):
                event = {**event, "utc": datetime.now(timezone.utc).isoformat(),
                         "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                line = json.dumps(event, sort_keys=True, allow_nan=False)
                log.write(line + "\n")
                log.flush()
                os.fsync(log.fileno())
                print(line, flush=True)
            journal = None
            try:
                snapshot = Path(manifest["snapshot"])
                source_digest = verify_source(snapshot, manifest["source_attestation"])
                attestation = read_attestation(manifest["input_attestation"])
                records, hashes = prepare_inputs(snapshot, manifest["corpus"], attestation)
                if torch.cuda.device_count() != 2:
                    raise ValueError("runtime must expose exactly the two RTX GPUs")
                for slot in manifest["coordinator_slots"]:
                    properties = torch.cuda.get_device_properties(slot["device"])
                    if str(properties.uuid) != slot["gpu_uuid"] or "RTX" not in properties.name:
                        raise ValueError("coordinator GPU identity differs from qualified RTX slot")
                token_path = Path(manifest["token_file"])
                if token_path.stat().st_mode & 0o077:
                    raise ValueError("worker token file must be private")
                client = EXL3RemoteClient(endpoints=[RemoteEndpoint(**item) for item in manifest["endpoints"]],
                    coordinator_slots=[CoordinatorSlot(**item) for item in manifest["coordinator_slots"]],
                    token=token_path.read_bytes().strip(), timeout_seconds=600, max_attempts=1,
                    assignment_store_path=root / "search-assignments.json")
                for endpoint in client.endpoints:
                    client.qualify(endpoint)
                source = V41Source(snapshot)
                # Bind the complete manifest as well as caller provenance, so a
                # topology/path change cannot silently resume old work.
                identity = dict(runtime=manifest, input_attestation=attestation, source_manifest_sha256=source_digest)
                journal = RunStore(root, identity)
                search = DistributedSearch(source, client, identity)
                driver = BlockDriver(source, journal, identity, device="cuda:0", search=search, progress=progress)
                sys.path.insert(0, str(snapshot / "inference"))
                kernels = importlib.import_module("kernel")
                if Path(kernels.__file__).resolve() != (snapshot / "inference/kernel.py").resolve():
                    raise ValueError("native kernel import did not resolve to pinned checkpoint")
                def main_adapters(guarded_kernels):
                    adapters = []
                    try:
                        for device in ("cuda:0", "cuda:1"):
                            adapters.append(source.load_main_input(copy.deepcopy(hashes), device))
                        return adapters
                    except BaseException:
                        for adapter in adapters:
                            adapter.close()
                        raise
                def draft_adapters(guarded_kernels):
                    return [source.load_dspark_input(device, native_kernels=guarded_kernels)
                            for device in ("cuda:0", "cuda:1")]
                progress(dict(event="runtime_inputs_ready", records=len(records),
                              tokens=sum(len(row["input_ids"]) for row in records)))
                quantize_namespaces(driver, records, attestation, main_adapters=main_adapters,
                                    draft_adapters=draft_adapters, native_kernels=kernels, replica_device="cuda:1")
                inventory = build_inventory(driver)
                metadata = model_metadata(source.config, inventory, provenance=manifest["identity"])
                assets = plan_assets(snapshot, json.loads(Path(manifest["source_attestation"]).read_text()))
                assets["README.md"] = model_card_asset(snapshot.name)
                result = write_export(inventory, manifest["output"], manifest["export_state"], identity=identity,
                                      resume=resume, metadata=metadata, assets=assets, progress=progress)
                validation = validate_export(manifest["output"], manifest["export_state"], snapshot)
                from write_export import _publish_json
                _publish_json(Path(manifest["export_state"]) / "structure-validation.json", validation)
                progress(dict(event="export_structure_validated", **validation))
                progress(dict(event="runtime_stage_complete", **result))
                return result
            except BaseException as error:
                progress(dict(event="runtime_failed", error_type=type(error).__name__, message=str(error)))
                raise
            finally:
                if journal is not None:
                    journal.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text()), resume=args.resume)
