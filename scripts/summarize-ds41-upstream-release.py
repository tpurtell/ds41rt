#!/usr/bin/env python3
"""Summarize the complete upstream-integration performance matrix and provenance."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
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
        "startup_seconds": {},
    }
    artifacts = set()
    binary_hashes = set()
    context_hashes = set()
    corpus_hashes = set()
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
                path = Path(cmd[cmd.index("--output") + 1])
                observed.append(path.name)
                assert path.resolve().parent == args.input.resolve()
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
            report["provenance"][f"{layout}-{phase}"] = metadata
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
            assert len(result[key]["cases"]) == 9
            assert all(case["samples"] == 3 for case in result[key]["cases"].values())
        assert result["retained_decode"]["contexts"] == [0, 32768, 65536, 131072, 262144]
        assert result["retained_decode_2k"]["contexts"] == [2048]
        assert result["prefill"]["bases"] == [0, 32768, 65536, 131072, 262144]
        assert result["prefill"]["suffixes"] == [1024, 2048, 4096, 8192, 16384, 32768]
        report["layouts"][layout] = result
    assert len(binary_hashes) == len(context_hashes) == len(corpus_hashes) == 1
    report["binary_sha256"] = next(iter(binary_hashes))
    report["context_sha256"] = next(iter(context_hashes))
    report["corpus_sha256"] = next(iter(corpus_hashes))
    report["artifacts"] = [helpers["artifact"](p) for p in sorted(artifacts)]
    report["performance_matrix_passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
