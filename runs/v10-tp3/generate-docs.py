#!/usr/bin/env python3
"""Generate the v10 TP3 campaign manifests and docs from one canonical source.

The three report documents are rendered from manifests, never hand-written:

  docs/release-v10-tp3-official-1x-3spark.md
  docs/release-v10-tp3-exl3-compact-1x-3spark.md
  docs/release-v10-tp3-campaign-status.md

Every family stays `PENDING` until raw evidence is named in the per-arm
manifests under `runs/v10-tp3/`. This script does not invent measurements and
never launches anything; it only writes manifests and invokes the CPU-only
renderer. Re-run it after raw evidence arrives and the same documents will carry
the measured values.

Usage:
  runs/v10-tp3/generate-docs.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
RENDERER = REPO / "scripts" / "render-ds41-v10-tp3-reports.py"
EXPECTED_COUNTS = REPO / "runs" / "v10-exl3-tp3" / "expected-counts.json"

STATUS = "NOT EXECUTED - pending raw manifests; no performance, memory, quality or readiness claim"

ARMS = (
    {
        "id": "v10-native-tp3ep1",
        "doc": "docs/release-v10-tp3-official-1x-3spark.md",
        "title": "DS41RT v10 TP3 official native 1x / 3-Spark: performance and qualification",
        "arm": {
            "id": "v10-native-tp3ep1",
            "quant": "official",
            "topology": "TP3xEP1",
            "topology_form": "explicit",
            "rtx": 1,
            "sparks": 3,
            "config_path": "examples/configs/tp3ep1-native.config",
            "comparison_group": "v10-1rtx-3spark",
            "resident_bytes_per_rank": 2406481920,
            "remote_layers": 35,
            "worker_allocated_layers": None,
            "weight_note": (
                "TP3 per-rank shard is the exact 2304/3 = 768-column slice, 2,406,481,920 B. "
                "Five RTX-local layers leave 35 remote: 35 x 2,406,481,920 = 84,226,867,200 B, "
                "inside the 109,119,320,064 B tested fleet floor by 24,892,452,864 B. "
                "Admission arithmetic only, not a runtime-fit claim."
            ),
            "ceiling": {
                "memory_reservation": None,
                "kv_pool_size": None,
                "prefill_batch_tokens": None,
                "note": "Uncapped native arm; no 32 GiB simulation applies.",
            },
            "runtime_flags": {
                "spark_count": 3, "spark_tp": 3, "spark_ep": 1,
                "rtx_expert_layers": 5, "remote_dispatch_layers": 35,
                "expert_format": "native", "dspark": "on",
            },
            "raw": {},
            "warmup": "one shape warmup, excluded from the ledger",
            "repeats": 3,
            "notes": "Not launched: the v10 image pair is not published, so no raw evidence exists and every family is PENDING.",
            "failures": [],
        },
    },
    {
        "id": "v10-exl3-compact-tp3",
        "doc": "docs/release-v10-tp3-exl3-compact-1x-3spark.md",
        "title": "DS41RT v10 compact EXL3 TP3 1x / 3-Spark: performance and qualification",
        "arm": {
            "id": "v10-exl3-compact-tp3",
            "quant": "exl3",
            "topology": "TP3",
            "topology_form": "implicit",
            "rtx": 1,
            "sparks": 3,
            "config_path": "examples/configs/exl3-compact-tp3.config",
            "comparison_group": "v10-1rtx-3spark",
            "resident_bytes_per_rank": None,
            "remote_layers": None,
            "worker_allocated_layers": None,
            "weight_note": (
                "Compact EXL3 TP3 is the implicit layout (SPARK_COUNT=3 with no "
                "SPARK_TP/SPARK_EP); each rank serves the 2304/3 = 768-column shard from "
                "non-paired disjoint k34-family packages. Resident bytes are not yet captured "
                "and are not inferred from the native arm."
            ),
            "ceiling": {
                "memory_reservation": "32GiB",
                "memory_reservation_bytes": 34359738368,
                "kv_pool_size": "2GiB",
                "kv_pool_size_bytes": 2147483648,
                "prefill_batch_tokens": 256,
                "evidence": None,
                "note": (
                    "Declared ceiling from examples/configs/exl3-compact-tp3.config. The resolved "
                    "reservation, occupancy and KV figures must be captured from the service at "
                    "launch; the declared values are not a proven fit."
                ),
            },
            "runtime_flags": {
                "spark_count": 3, "spark_tp": "implicit (count = TP)", "spark_ep": 1,
                "expert_format": "exl3", "exl3_family": "k34",
                "memory_reservation": "32GiB", "kv_pool_size": "2GiB",
                "prefill_batch_tokens": 256, "dspark": "on",
            },
            "raw": {},
            "warmup": "one shape warmup, excluded from the ledger",
            "repeats": 3,
            "notes": (
                "Not launched: the v10 image pair is not published and the k34 "
                "tp3-rank{0,1,2} packages must be present in the Spark image. Every family is PENDING."
            ),
            "failures": [],
        },
    },
)


def base_manifest() -> dict:
    return {
        "release": "v10",
        "checkpoint": {
            "model_id": "deepseek-ai/DeepSeek-V4.1-Flash",
            "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "quant": "official",
        },
        "accounting": {
            "repeats": 3,
            "prefill_warmups": 1,
            "tool_eval_runs": 3,
            "note": ("One shape warmup then three timed samples per cell; retained uses three "
                     "repeats per isolated context. Expected counts are re-derived from the "
                     "live harness scripts at check/render time."),
        },
        "report_status": STATUS,
    }


def build() -> dict:
    base = base_manifest()
    documents = {}
    for entry in ARMS:
        arm = json.loads(json.dumps(entry["arm"]))
        config = REPO / arm["config_path"]
        arm["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        arm["notes"] += (f" config_sha256 is the current committed hash of {arm['config_path']} "
                         "(re-verify at freeze).")
        manifest = {k: v for k, v in base.items() if k != "arms"}
        manifest["arms"] = [arm]
        manifest["report_title"] = entry["title"]
        documents[f"runs/v10-tp3/{entry['id']}-manifest.json"] = manifest
        documents[entry["doc"]] = None  # rendered, not written directly
    status = {k: v for k, v in base.items() if k != "arms"}
    status["arms"] = [json.loads(json.dumps(entry["arm"])) for entry in ARMS]
    for arm, entry in zip(status["arms"], ARMS):
        arm["config_sha256"] = documents[f"runs/v10-tp3/{entry['id']}-manifest.json"]["arms"][0]["config_sha256"]
    status["report_title"] = "DS41RT v10 TP3 campaign status: native TP3EP1 and compact EXL3 TP3"
    documents["runs/v10-tp3/campaign-manifest.json"] = status
    documents["docs/release-v10-tp3-campaign-status.md"] = None
    return documents


def write(documents: dict, check: bool) -> int:
    problems = []
    for relative, document in documents.items():
        if document is None:
            continue
        path = REPO / relative
        text = json.dumps(document, indent=2) + "\n"
        if check:
            if not path.is_file() or path.read_text() != text:
                problems.append(f"{relative} is stale")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {relative}")

    render_jobs = [(entry["id"], entry["doc"]) for entry in ARMS]
    render_jobs.append((None, "docs/release-v10-tp3-campaign-status.md"))
    for arm_id, doc in render_jobs:
        manifest = (f"runs/v10-tp3/{arm_id}-manifest.json" if arm_id
                    else "runs/v10-tp3/campaign-manifest.json")
        if arm_id is not None:
            ids = [a["id"] for a in json.loads((REPO / manifest).read_text())["arms"]]
            if ids != [arm_id]:
                problems.append(f"{doc}: manifest {manifest} declares {ids}, expected [{arm_id}]")
                continue
        base_command = [sys.executable, str(RENDERER), "--manifest", manifest,
                        "--package", "runs/v10-tp3"]
        if EXPECTED_COUNTS.is_file():
            base_command += ["--expected-counts", str(EXPECTED_COUNTS.relative_to(REPO))]
        if check:
            result = subprocess.run(base_command + ["--check"], cwd=REPO,
                                    capture_output=True, text=True)
            if result.returncode not in (0, 2):  # 2 == pending families, expected here
                problems.append(f"{doc}: renderer check rc={result.returncode}: {result.stderr}")
            continue
        result = subprocess.run(base_command + ["--output", doc], cwd=REPO,
                                capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"{doc}: render rc={result.returncode}: {result.stderr}")
            continue
        print(f"wrote {doc}")
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Do not write; report stale manifests/docs.")
    args = parser.parse_args()
    return write(build(), args.check)


if __name__ == "__main__":
    raise SystemExit(main())
