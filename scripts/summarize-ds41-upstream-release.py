#!/usr/bin/env python3
"""Summarize the complete upstream-integration performance matrix and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import runpy
from pathlib import Path


def validate_decode_corpus(report: dict, corpus: dict) -> None:
    """Require the actual requests, including reasoning controls, to match."""
    expected = set(corpus["weighted_case_ids"]) | {"counting"}
    samples = report["samples"]
    assert {sample["case"] for sample in samples} == expected
    for case in expected:
        selected = [sample for sample in samples if sample["case"] == case]
        assert len(selected) == 3 and {sample["repeat"] for sample in selected} == {1, 2, 3}
        definition = corpus["cases"][case] if case != "counting" else {"thinking": "disabled"}
        for sample in selected:
            request = sample["request"]
            assert request["thinking"] == {"type": definition.get("thinking", "disabled")}
            assert request.get("reasoning_effort") == definition.get("reasoning_effort")


def summarize_readiness(campaign: dict, build: dict) -> dict:
    """Use post-readiness snapshots from the standard launches, in GPU order."""
    assert campaign.get("measurements_passed") is True and campaign.get("completed_ns")
    result = {}
    for layout, label, count in (("single", "single-dspark", 1), ("dual", "dual-final", 2)):
        events = [event for event in campaign["events"] if event.get("label") == label]
        assert len(events) == 1
        event = events[0]
        assert event["exit_code"] == 0
        container = event["container"]
        assert container["Image"] == build["coordinator"]["image_id"]
        assert container["State"]["Running"]
        assert "--dspark" in container["Config"]["Cmd"]
        ids = [uuid for request in container["HostConfig"]["DeviceRequests"]
               for uuid in request["DeviceIDs"]]
        assert len(ids) == len(set(ids)) == count
        rows = {row["uuid"]: row for row in csv.DictReader(
            io.StringIO(event["gpu_memory"]), skipinitialspace=True)}
        gpus = []
        for uuid in ids:
            row = rows[uuid]
            assert float(row["power.limit [W]"].split()[0]) == 400
            gpus.append({"uuid": uuid,
                         "used_mib": float(row["memory.used [MiB]"].split()[0]),
                         "free_mib": float(row["memory.free [MiB]"].split()[0])})
        result[layout] = {"launch_label": label, "gpus": gpus}
    return result


def summarize_telemetry(path: Path) -> dict:
    """Keep controls and memory peaks per monitored GPU, including idle cards."""
    gpus = {}
    columns = {
        "peak_memory_mib": "memory.used [MiB]",
        "peak_power_watts": "power.draw [W]",
        "peak_memory_clock_mhz": "clocks.current.memory [MHz]",
        "peak_temperature_celsius": "temperature.gpu",
    }
    with path.open() as source:
        for row in csv.DictReader(source, skipinitialspace=True):
            assert float(row["power.limit [W]"].split()[0]) == 400, path
            gpu = gpus.setdefault(row["uuid"], {"samples": 0, "power_limit_watts": 400})
            gpu["samples"] += 1
            free = float(row["memory.free [MiB]"].split()[0])
            gpu["minimum_free_mib"] = min(gpu.get("minimum_free_mib", free), free)
            for output, column in columns.items():
                value = float(row[column].split()[0])
                gpu[output] = max(gpu.get(output, value), value)
            assert gpu["peak_memory_clock_mhz"] <= 14001, path
    assert gpus, path
    return gpus


def summarize_deployment(metadata: dict, log: Path) -> dict:
    """Read measured pool/placement and explicit launch controls."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    argv = metadata["arguments"]
    def option(args, name):
        return int(args[args.index(name) + 1])
    concurrency = option(argv, "--concurrency")
    retained = option(argv, "--prefix-cache-entries")
    rows = [line for line in text.splitlines() if
            "native KV pool reservation" in line or
            "dual RTX cache reservation after fixed allocations" in line]
    assert len(rows) == 1, log
    line = rows[0]
    pages = json.loads(re.search(r"source_pages=(\[[^]]+\])", line).group(1))
    pool_bytes = int(re.search(r"global_bytes=(\d+)", line).group(1))
    runtime_headroom = int(re.search(r"runtime_headroom_bytes=(\d+)", line).group(1))
    tails = concurrency + 2 * retained
    assert len(pages) == 4 and pages == [pages[0]] * 3 + [2 * pages[0]]
    assert pages[0] > tails
    if metadata["layout"] == "dual":
        layers = int(re.search(r"(?:rtx_expert_layers|encoder_layers)=(\d+)", line).group(1))
        tp = 2
    else:
        placement = next(line for line in text.splitlines() if "bottom-up RTX expert placement" in line)
        layers = int(re.search(r"layers=(\d+)", placement).group(1))
        tp = 1
    workers = metadata["workers"]
    budgets = {option(w["args"], "--device-budget-bytes") for w in workers}
    first = {option(w["args"], "--first-layer") for w in workers}
    assert len(budgets) == len(first) == 1
    spark_first = next(iter(first))
    assert 0 <= spark_first <= layers <= 40, "RTX/Spark expert partition has a gap"
    return {"global_pool_bytes": pool_bytes, "logical_pool_tokens": (pages[0] - tails) * 512,
            "runtime_headroom_bytes_per_gpu": runtime_headroom,
            "private_tail_tokens": tails * 512, "source_pages": pages,
            "prompt_retention_entries": retained, "completed_turn_retention_entries": retained,
            "concurrency": concurrency, "rtx_expert_layers": layers, "rtx_expert_tp": tp,
            "spark_first_layer": spark_first, "spark_layers": 40 - spark_first,
            "spark_active_layers": 40 - layers, "spark_redundant_layers": layers - spark_first,
            "spark_budget_bytes_each": next(iter(budgets))}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path,
                        help="Explicit release corpus; omit to reproduce the historical eight-category report")
    parser.add_argument("--model", choices=("full", "exl3"), default="full")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    historical_cases = ["code", "math", "fable", "hello", "topic", "structured-json",
                        "structured-json-schema", "multilingual"]
    corpus = json.loads(args.corpus.read_text()) if args.corpus else None
    weighted_cases = corpus["weighted_case_ids"] if corpus else historical_cases
    assert len(weighted_cases) == len(set(weighted_cases)) and "counting" not in weighted_cases
    expected_cases = set(weighted_cases) | {"counting"}
    required = [
        args.input / f"{layout}-{phase}-metadata.record"
        for layout in ("single", "dual")
        for phase in ("dspark", "target", "retained_2k")
    ] + [args.input / f"{layout}-startup-seconds.txt" for layout in ("single", "dual")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        parser.error("matrix incomplete; missing: " + ", ".join(missing))
    helpers = runpy.run_path(
        str(Path(__file__).with_name("summarize-ds41-phase2-release.py"))
    )
    load = helpers["load"]
    build = load(args.build_record)
    assert build["build_exit_code"] == build["launch_exit_code"] == 0
    worker_images = {worker["image_id"] for worker in build["workers"]}
    assert len(build["workers"]) == 4 and len(worker_images) == 1
    report = {
        "schema": 1,
        "scope": "Upstream-integration release performance; quality evidence is separate",
        "model": args.model,
        "weighted_case_ids": weighted_cases,
        "engine_commit": build["source"]["engine_revision"],
        "source_manifest_sha256": build["source"]["manifest_sha256"],
        "build_record": helpers["artifact"](args.build_record),
        "controls": {
            "rtx_power_limit_watts": 400,
            "rtx_memory": "stock; loaded 13,365 MHz, maximum 14,001 MHz",
            "temperature": 0,
            "thinking": "disabled",
            "repeats": 3,
        },
        "layouts": {},
        "provenance": {},
        "gpu_telemetry": {},
        "startup_seconds": {},
        "deployment": {},
    }
    artifacts = set()
    if corpus:
        report["controls"]["thinking"] = "per-case"
        report["controls"]["case_controls"] = {
            case: {"thinking": corpus["cases"][case].get("thinking", "disabled"),
                   "reasoning_effort": corpus["cases"][case].get("reasoning_effort"),
                   "weight": corpus["cases"][case]["weight"]}
            for case in weighted_cases}
        report["corpus_artifact"] = helpers["artifact"](args.corpus)
    binary_hashes = set()
    context_hashes = set()
    corpus_hashes = set()
    recorded_output_directories = set()
    for layout in ("single", "dual"):
        expected = {
            "dspark": {
                f"{layout}-decode.json", f"{layout}-prefill.json",
                f"{layout}-retained-decode.json",
                *(f"{layout}-{case}.json" for case in ("code", "topic", "counting")),
                *(f"{layout}-mixed-{i}.json" for i in range(1, 4)),
            },
            "target": {f"{layout}-target-decode.json"},
            "retained_2k": {f"{layout}-retained-2k.json"},
        }
        for phase, names in expected.items():
            metadata_path = args.input / f"{layout}-{phase}-metadata.record"
            metadata = load(metadata_path)
            assert metadata["layout"] == layout and metadata["phase"] == phase
            assert metadata.get("completed_ns"), metadata_path
            assert metadata["image"] == build["coordinator"]["image_id"]
            assert len(metadata["workers"]) == 4
            assert {w["image"] for w in metadata["workers"]} == worker_images
            argv = metadata["arguments"]
            assert argv[argv.index("--rtx-gpus") + 1] == ("1" if layout == "single" else "2")
            assert ("--dspark" in argv) == (phase != "target")
            binary_hashes.add(metadata["binary_sha256"])
            context_hashes.add(metadata["source_sha256"])
            observed = []
            for command in metadata["commands"]:
                assert command["exit_code"] == 0
                cmd = command["command"]
                recorded_path = Path(cmd[cmd.index("--output") + 1])
                recorded_output_directories.add(str(recorded_path.parent))
                path = args.input / recorded_path.name
                observed.append(path.name)
                raw = load(path)
                assert raw["passed"] is True, path
                if "corpus_sha256" in raw:
                    corpus_hashes.add(raw["corpus_sha256"])
                artifacts.add(path)
            assert set(observed) == names and len(observed) == len(names), metadata_path
            artifacts.add(metadata_path)
            telemetry = args.input / f"{layout}-{phase}-gpu.csv"
            assert telemetry.is_file() and telemetry.stat().st_size > 0
            artifacts.add(telemetry)
            report["gpu_telemetry"][f"{layout}-{phase}"] = summarize_telemetry(telemetry)
            report["provenance"][f"{layout}-{phase}"] = metadata
            if phase == "dspark":
                log = args.input / f"{layout}-{phase}-server.log"
                report["deployment"][layout] = summarize_deployment(metadata, log)
                artifacts.add(log)
        startup = args.input / f"{layout}-startup-seconds.txt"
        seconds = float(startup.read_text())
        assert seconds > 0
        report["startup_seconds"][layout] = seconds
        artifacts.add(startup)
        result = {
            "rtx_gpus": 1 if layout == "single" else 2,
            "decode": helpers["summarize_decode"](args.input / f"{layout}-decode.json"),
            "target_decode": helpers["summarize_decode"](args.input / f"{layout}-target-decode.json"),
            "concurrency": {
                case: helpers["summarize_concurrency"](args.input / f"{layout}-{case}.json")
                for case in ("counting", "code", "topic")
            },
            "mixed": helpers["summarize_mixed"]([
                args.input / f"{layout}-mixed-{i}.json" for i in range(1, 4)
            ]),
            "prefill": helpers["summarize_prefill"](args.input / f"{layout}-prefill.json"),
            "retained_decode": helpers["summarize_retained"](args.input / f"{layout}-retained-decode.json"),
            "retained_decode_2k": helpers["summarize_retained"](args.input / f"{layout}-retained-2k.json"),
        }
        for key in ("decode", "target_decode"):
            assert result[key]["passed"] and result[key]["repeats"] == 3
            assert set(result[key]["cases"]) == expected_cases, "release content cases differ from corpus"
            assert all(case["samples"] == 3 for case in result[key]["cases"].values())
            if corpus:
                filename = f"{layout}-{'target-' if key == 'target_decode' else ''}decode.json"
                validate_decode_corpus(load(args.input / filename), corpus)
        assert result["retained_decode"]["contexts"] == [0, 32768, 65536, 131072, 262144]
        assert result["retained_decode_2k"]["contexts"] == [2048]
        assert result["prefill"]["bases"] == [0, 32768, 65536, 131072, 262144]
        assert result["prefill"]["suffixes"] == [1024, 2048, 4096, 8192, 16384, 32768]
        report["layouts"][layout] = result
    assert len(binary_hashes) == len(context_hashes) == len(corpus_hashes) == 1
    assert len(recorded_output_directories) == 1
    report["binary_sha256"] = next(iter(binary_hashes))
    report["context_sha256"] = next(iter(context_hashes))
    report["corpus_sha256"] = next(iter(corpus_hashes))
    if args.corpus:
        assert report["corpus_sha256"] == hashlib.sha256(args.corpus.read_bytes()).hexdigest(), \
            "measured corpus differs from requested release corpus"
    campaign_path = args.input / "campaign.record"
    campaign = load(campaign_path)
    report["readiness_memory"] = summarize_readiness(campaign, build)
    artifacts.add(campaign_path)
    if previous := campaign.get("previous_attempt"):
        assert Path(previous).name == previous
        prior = args.input / previous
        assert load(prior).get("completed_ns")
        report["previous_campaign_attempt"] = helpers["artifact"](prior)
        artifacts.add(prior)
        artifacts.add(prior.with_suffix(".log"))
    report["artifacts"] = [helpers["artifact"](p) for p in sorted(artifacts)]
    report["performance_matrix_passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
