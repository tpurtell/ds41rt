"""Explicit compatible execution repairs without rewriting saved provenance."""
import copy
import hashlib
import json
from pathlib import Path

from gptqmodel.utils import v41_checkpoint


def _bounded(path, root, limit):
    path = Path(path)
    if not path.is_absolute() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > limit:
        raise ValueError("recovery evidence must be a bounded run-local file")
    return path.read_bytes()


def resolve_identity(manifest):
    recovery = manifest.get("recovery")
    if recovery is None:
        return manifest, None
    if recovery.get("schema") == "ds41rt-continuous-search-recovery-v1":
        root = Path(manifest["run_root"])
        previous = json.loads(_bounded(recovery["previous_manifest"], root, 1024 * 1024))
        if previous.get("recovery", {}).get("schema") != "ds41rt-input-serializer-recovery-v1":
            raise ValueError("continuous search recovery requires the prior serializer execution")
        original, previous_evidence = resolve_identity(previous)
        normalized = copy.deepcopy(manifest)
        normalized["recovery"] = previous["recovery"]
        for old, new in zip(previous["coordinator_slots"], normalized["coordinator_slots"], strict=True):
            if {k: v for k, v in old.items() if k not in {"image_digest", "preflight_sha256"}} != {
                    k: v for k, v in new.items() if k not in {"image_digest", "preflight_sha256"}}:
                raise ValueError("search recovery cannot change coordinator GPU topology")
        normalized["coordinator_slots"] = previous["coordinator_slots"]
        if normalized != previous:
            raise ValueError("search recovery changed model/corpus/recipe/runtime inputs")
        payload = _bounded(recovery["search_report"], root, 4 * 1024 * 1024)
        if hashlib.sha256(payload).hexdigest() != recovery["search_report_sha256"]:
            raise ValueError("search recovery evidence checksum differs")
        reports = [json.loads(line) for line in payload.decode().splitlines()
                   if line.startswith('{"event": "throughput_passed"')]
        passed = [item for item in reports if item.get("variant") == "continuous"]
        if (len(passed) != 1 or passed[0].get("packed_exact") is not True
                or passed[0].get("jobs", 0) < 64
                or set(passed[0].get("devices", {})) != {"cuda:0", "cuda:1", "ostrich", "dodo", "emu", "kiwi"}
                or any(count <= 0 for count in passed[0]["devices"].values())):
            raise ValueError("search recovery lacks exact six-device qualification")
        return original, dict(schema=recovery["schema"], execution_manifest=manifest,
            previous_evidence=previous_evidence, search_report_sha256=recovery["search_report_sha256"])
    if recovery.get("schema") != "ds41rt-input-serializer-recovery-v1":
        raise ValueError("unsupported recovery authorization")
    root = Path(manifest["run_root"])
    original = json.loads(_bounded(recovery["original_manifest"], root, 1024 * 1024))
    if "recovery" in original:
        raise ValueError("nested recovery identities are not supported")
    normalized = copy.deepcopy(manifest)
    del normalized["recovery"]
    for old, new in zip(original["coordinator_slots"], normalized["coordinator_slots"], strict=True):
        if {k: v for k, v in old.items() if k not in {"image_digest", "preflight_sha256"}} != {
                k: v for k, v in new.items() if k not in {"image_digest", "preflight_sha256"}}:
            raise ValueError("serializer recovery cannot change coordinator GPU topology")
    normalized["coordinator_slots"] = original["coordinator_slots"]
    if normalized != original:
        raise ValueError("serializer recovery changed model/corpus/recipe/runtime inputs")
    payload = _bounded(recovery["memory_report"], root, 4 * 1024 * 1024)
    if hashlib.sha256(payload).hexdigest() != recovery["memory_report_sha256"]:
        raise ValueError("recovery memory evidence checksum differs")
    reports = [json.loads(line) for line in payload.decode().splitlines() if line.startswith('{"event": "checkpoint_memory_probe_passed"')]
    if len(reports) != 1:
        raise ValueError("recovery requires one passed memory/byte-compatibility probe")
    report = reports[0]
    serializer_sha = hashlib.sha256(Path(v41_checkpoint.__file__).read_bytes()).hexdigest()
    if (report.get("serializer_sha256") != serializer_sha or report.get("gc_disabled") is not True
            or report.get("workers") != 2 or report.get("waves", 0) < 32
            or not 0 <= report.get("post_warmup_spread_kib", -1) <= 256 * 1024
            or len(set(report.get("byte_exact_inputs", []))) != 3):
        raise ValueError("recovery lacks qualified serializer release/compatibility evidence")
    return original, dict(schema=recovery["schema"], execution_manifest=manifest,
                         memory_report_sha256=recovery["memory_report_sha256"], serializer_sha256=serializer_sha)


def authorize_input_recovery(driver, evidence):
    if evidence is None:
        return
    if evidence["schema"] == "ds41rt-continuous-search-recovery-v1":
        previous_key = "recovery/input-serializer-release-v1"
        if driver._load(previous_key, "recovery") != evidence["previous_evidence"]:
            raise ValueError("search recovery differs from committed prior execution")
        key = "recovery/continuous-search-v1"
        existing = driver._load(key, "recovery")
        if existing is not None:
            if existing != evidence:
                raise ValueError("committed search recovery execution identity changed")
            return
        if (driver.journal.root / "search-assignments-continuous-v1.json").exists():
            raise ValueError("new search epoch exists without authorization")
        driver._publish(key, "recovery", evidence, (previous_key,))
        driver.progress(dict(event="continuous_search_recovery_authorized",
            committed_candidates="reused", prior_assignments="preserved", unfinished_searches="recompute_in_new_epoch"))
        return
    key = "recovery/input-serializer-release-v1"
    existing = driver._load(key, "recovery")
    if existing is not None:
        if existing != evidence:
            raise ValueError("committed recovery execution identity changed")
        return
    journal = driver.journal
    if journal.db.execute("SELECT count(*) FROM artifacts WHERE key LIKE 'blocks/%'").fetchone()[0]:
        raise ValueError("serializer recovery authorization requires an input-only boundary")
    if (journal.root / "search-assignments.json").exists():
        raise ValueError("serializer recovery cannot migrate existing search assignments")
    corpus = driver._load("inputs/inventory", "inventory")
    count = journal.db.execute("SELECT count(*) FROM artifacts WHERE key LIKE 'inputs/frontiers/%'").fetchone()[0]
    if corpus is None or not count or count != len(corpus["records"]):
        raise ValueError("serializer recovery requires the complete committed input inventory")
    # Existing payloads keep their original provenance. The new execution
    # identity is explicitly journaled; search metrics also report the new slots.
    driver._publish(key, "recovery", evidence, ("inputs/inventory",))
    driver.progress(dict(event="input_serializer_recovery_authorized", inputs=count,
                         serializer_sha256=evidence["serializer_sha256"]))
