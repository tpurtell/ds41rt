#!/usr/bin/env python3
"""Parse matched TP×EP decode-profile logs into validated per-rank metrics.

CPU-only analysis support. Consumes the DEBUG lines emitted by a profile run:

    coordinator: RUST_LOG=info,ds41rt::timing=debug,ds41rt::route_policy=debug
                 DS41RT_PROTOCOL_V2_TCP_TIMING=1
    worker:      RUST_LOG=info,ds41rt::expert_timing=debug DS41RT_PROTOCOL_V2_TCP_TIMING=1

Validated semantics (traced against the emitters, not assumed):
- Worker `kernel_us`/`compact_us` are normalized by logical work
  `owned_rows * (2304 // tp)`, with a kernel-extent variant (640 for TP4) to
  expose padding. Raw per-owned-route kernel time is never compared across
  different TP shard widths.
- Worker/coordinator timing lines carry no request id, so worker aggregates are
  a **worst-rank median over a (layer, rows) bucket**, not a per-request
  critical path.
- `expert_rows_histogram` is a binned expert-row-count distribution (bins 1..16
  plus a 17+ tail with `expert_rows_tail_routes`), not an expert-id list.
- dSpark round (`native independent lane round` / `native scheduler round`):
  `accepted` is `Σ accepted_inputs` (includes the per-request anchor) and
  `emitted` is `Σ` newly generated tokens, so
  `drafts_accepted = accepted - requests`,
  `acceptance_ratio = drafts_accepted / proposed`,
  `tokens_per_round = emitted/rounds`.
  `accepted + emitted` would double count and is never computed.
- Transport lines carry request ids, but ids are per-side counters (not proven
  globally shared), so the honest fallback is a **per-rank** observation set.
  A cross-rank join is only produced when `--run-id` and
  `--assume-global-request-id` are both supplied, and it is labelled as an
  unverified assumption (per-side counters are not proven global).

Raw records are retained; aggregates are medians/p95/min/max.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")
NUMBER = r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"
FIELD = re.compile(rf"\b([A-Za-z_]\w*)=({NUMBER})\b")
HISTOGRAM = re.compile(r"expert_rows_histogram=\[([0-9, ]+)\]")
INT_KEYS = ("executor_id", "layer", "rows", "active_experts", "max_expert_rows",
            "kernel_capacity", "lane", "requests", "proposed", "accepted", "emitted",
            "execution_lane", "request_id", "layer_id", "distinct_experts")


def fields(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in FIELD.findall(line)}


def integer(value: float | None, name: str) -> int | None:
    if value is None:
        return None
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"non-integral {name}={value!r}")
    return int(value)


def histogram(line: str) -> list[int]:
    match = HISTOGRAM.search(line)
    return [int(value) for value in match.group(1).split(",")] if match else []


def percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "median": statistics.median(ordered),
            "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)],
            "max": ordered[-1], "min": ordered[0]}


def shard_widths(tp: int) -> tuple[int, int]:
    logical = 2304 // tp
    return logical, 640 if tp == 4 else logical


def parse_line(source: str, line: str, tp: int) -> dict | None:
    ts_match = TIMESTAMP.match(line)
    ts = ts_match.group(1) if ts_match else None
    values = fields(line)
    ints = {key: integer(values.get(key), key) for key in INT_KEYS}

    if "native expert execution" in line:
        counts = histogram(line)
        owned_rows = sum((i + 1) * v for i, v in enumerate(counts[:16]))
        if len(counts) > 16:
            owned_rows += int(values.get("expert_rows_tail_routes", 0.0))
        logical, kernel = shard_widths(tp)
        has_rows = owned_rows > 0
        return {"kind": "worker", "source": source, "ts": ts, **ints,
                "owned_rows": owned_rows, "histogram": counts,
                "tail_routes": int(values.get("expert_rows_tail_routes", 0.0)),
                "kernel_us": values.get("kernel_us"), "compact_us": values.get("compact_us"),
                "upload_us": values.get("upload_us"),
                "execution_host_us": values.get("execution_host_us"),
                "download_us": values.get("download_us"), "total_us": values.get("total_us"),
                "output_bytes": values.get("output_bytes"),
                "unique_expert_weight_bytes": values.get("unique_expert_weight_bytes"),
                "kernel_us_per_logical_unit":
                    values["kernel_us"] / (owned_rows * logical) if has_rows and "kernel_us" in values else None,
                "kernel_us_per_kernel_unit":
                    values["kernel_us"] / (owned_rows * kernel) if has_rows and "kernel_us" in values else None,
                "compact_us_per_logical_unit":
                    values["compact_us"] / (owned_rows * logical) if has_rows and "compact_us" in values else None}
    if "roundtrip_timing" in line:
        return {"kind": "transport", "source": source, "ts": ts, **ints,
                "request_wire_bytes": values.get("request_wire_bytes"),
                "response_wire_bytes": values.get("response_wire_bytes"),
                "execute_ms": values.get("execute_ms"), "send_ms": values.get("send_ms"),
                "poll_recv_ms": values.get("poll_recv_ms"), "parse_ms": values.get("parse_ms"),
                "total_ms": values.get("total_ms")}
    if "ds41rt::timing" not in line:
        return None
    if "native independent lane round" in line or "native scheduler round" in line:
        kind = "lane_round"
    elif "target experts" in line:
        kind = "target_experts"
    elif "target collection" in line:
        kind = "target_collection"
    elif "target local experts" in line:
        kind = "target_local_experts"
    elif "distributed target layer" in line:
        kind = "distributed_target_layer"
    elif "distributed routed expert reuse" in line:
        kind = "routed_expert_reuse"
    elif "TP2 routed completion" in line:
        kind = "tp2_routed_completion"
    else:
        return None
    record = {"kind": kind, "source": source, "ts": ts, **ints}
    for key in ("routed_us", "dispatch_us", "shared_and_collect_us", "shared_us", "collect_us",
                "shared_copy_us", "upload_us", "receive_us", "reduce_us", "total_us",
                "production_and_index_us", "attention_us", "experts_us", "finish_us",
                "producer_us", "copy_reduce_us", "draft_us", "prepared_us", "prepare_us",
                "verify_us", "distinct_experts"):
        if key in values:
            record[key] = values[key]
    return record


def load(paths: list[Path], tp: int, rows_filter: set[int] | None,
         start: str | None, end: str | None) -> list[dict]:
    records = []
    for path in paths:
        for raw in path.read_text(errors="replace").splitlines():
            line = ANSI.sub("", raw).rstrip()
            record = parse_line(path.name, line, tp)
            if record is None:
                continue
            if rows_filter is not None and record.get("rows") is not None \
                    and record["rows"] not in rows_filter:
                # Lane-round records carry no row count and are kept when they
                # fall inside the requested time window.
                continue
            # Timestamp-less transport eprintln lines cannot be windowed; they
            # remain subject to the rows filter.
            if start is not None and record["ts"] is not None and record["ts"] < start:
                continue
            if end is not None and record["ts"] is not None and record["ts"] > end:
                continue
            records.append(record)
    return records


def aggregate_worker(workers: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for record in workers:
        groups.setdefault((record["source"], record["executor_id"], record["layer"],
                           record["rows"], record["kernel_capacity"]), []).append(record)
    rank_groups = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        source, executor_id, layer, rows, capacity = key
        rank_groups.append({
            "source": source, "executor_id": executor_id, "layer": layer,
            "rows": rows, "capacity": capacity, "samples": len(group),
            "active_experts": [r["active_experts"] for r in group],
            "owned_rows": [r["owned_rows"] for r in group],
            "kernel_us": percentiles([r["kernel_us"] for r in group if r["kernel_us"] is not None]),
            "compact_us": percentiles([r["compact_us"] for r in group if r["compact_us"] is not None]),
            "kernel_us_per_logical_unit": percentiles(
                [r["kernel_us_per_logical_unit"] for r in group if r["kernel_us_per_logical_unit"] is not None]),
            "kernel_us_per_kernel_unit": percentiles(
                [r["kernel_us_per_kernel_unit"] for r in group if r["kernel_us_per_kernel_unit"] is not None]),
        })

    buckets: dict[tuple, dict[int, dict[str, list[float]]]] = {}
    for record in workers:
        per_rank = buckets.setdefault((record["layer"], record["rows"]), {})
        bucket = per_rank.setdefault(record["executor_id"], {"kernel": [], "compact": [],
                                                             "logical": [], "active": []})
        if record["kernel_us"] is not None:
            bucket["kernel"].append(record["kernel_us"])
        if record["compact_us"] is not None:
            bucket["compact"].append(record["compact_us"])
        if record["kernel_us_per_logical_unit"] is not None:
            bucket["logical"].append(record["kernel_us_per_logical_unit"])
        if record["active_experts"] is not None:
            bucket["active"].append(float(record["active_experts"]))
    worst = []
    for (layer, rows), per_rank in sorted(buckets.items()):
        kernel = {r: statistics.median(v["kernel"]) for r, v in per_rank.items() if v["kernel"]}
        compact = {r: statistics.median(v["compact"]) for r, v in per_rank.items() if v["compact"]}
        logical = {r: statistics.median(v["logical"]) for r, v in per_rank.items() if v["logical"]}
        kw = max(kernel, key=kernel.get) if kernel else None
        cw = max(compact, key=compact.get) if compact else None
        lw = max(logical, key=logical.get) if logical else None
        worst.append({
            "layer": layer, "rows": rows, "ranks": len(set(kernel) | set(compact)),
            "kernel_worst_rank": kw,
            "worst_rank_median_kernel_us": kernel.get(kw) if kw is not None else None,
            "best_rank_median_kernel_us": min(kernel.values()) if kernel else None,
            "kernel_worst_rank_compact_us": compact.get(kw) if kw is not None else None,
            "compact_worst_rank": cw,
            "worst_rank_median_compact_us": compact.get(cw) if cw is not None else None,
            "logical_worst_rank": lw,
            "worst_rank_median_kernel_us_per_logical_unit": logical.get(lw) if lw is not None else None,
            "best_rank_median_kernel_us_per_logical_unit": min(logical.values()) if logical else None,
        })
    return {"rank_groups": rank_groups, "worst_rank_median": worst}


def aggregate_coordinator(records: list[dict]) -> dict:
    phases: dict[str, dict[tuple, list[float]]] = {}
    for record in records:
        if record["kind"] not in ("target_experts", "target_collection", "target_local_experts",
                                  "distributed_target_layer", "tp2_routed_completion"):
            continue
        key = (record.get("layer"), record.get("rows"))
        for name in ("routed_us", "dispatch_us", "shared_and_collect_us", "shared_us", "collect_us",
                     "shared_copy_us", "upload_us", "receive_us", "reduce_us", "total_us",
                     "production_and_index_us", "attention_us", "experts_us", "finish_us",
                     "producer_us", "copy_reduce_us"):
            if record.get(name) is not None:
                phases.setdefault(name, {}).setdefault(key, []).append(record[name])
    return {name: [{"layer": key[0], "rows": key[1], **percentiles(values)}
                   for key, values in sorted(groups.items())]
            for name, groups in sorted(phases.items())}


def aggregate_dspark(records: list[dict]) -> dict | None:
    rounds = [r for r in records if r["kind"] == "lane_round"]
    if not rounds:
        return None
    proposed = sum(r["proposed"] or 0 for r in rounds)
    accepted = sum(r["accepted"] or 0 for r in rounds)
    emitted = sum(r["emitted"] or 0 for r in rounds)
    requests = sum(r["requests"] or 0 for r in rounds)
    drafts_accepted = accepted - requests
    return {
        "rounds": len(rounds), "requests": requests,
        "proposed_drafts": proposed, "accepted_inputs_incl_anchor": accepted,
        "emitted_tokens": emitted,
        "drafts_accepted": drafts_accepted,
        "acceptance_ratio": drafts_accepted / proposed if proposed else None,
        # emitted already includes every new token; accepted+emitted would double count.
        "tokens_per_round": emitted / len(rounds),
        "requests_per_round": requests / len(rounds),
        "draft_us": percentiles([r["draft_us"] for r in rounds if r.get("draft_us") is not None]),
        "verify_us": percentiles([r["verify_us"] for r in rounds if r.get("verify_us") is not None]),
        "prepared_us": percentiles([(r.get("prepared_us") if r.get("prepared_us") is not None
                                     else r.get("prepare_us")) for r in rounds
                                    if (r.get("prepared_us") is not None or r.get("prepare_us") is not None)]),
        "total_us": percentiles([r["total_us"] for r in rounds if r.get("total_us") is not None]),
    }


def aggregate_transport(records: list[dict], run_id: str | None,
                        global_ids: bool) -> tuple[dict, int]:
    if run_id and global_ids:
        groups: dict[tuple, list[dict]] = {}
        dropped = 0
        for record in records:
            if record.get("request_id") is None or record.get("layer_id") is None:
                dropped += 1
                continue
            groups.setdefault((run_id, record["request_id"], record["layer_id"], record["rows"]), []).append(record)
        joined = []
        for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
            latencies = {r["source"]: r["total_ms"] for r in group if r["total_ms"] is not None}
            if latencies:
                worst = max(latencies, key=latencies.get)
                joined.append({"run_id": key[0], "request_id": key[1], "layer_id": key[2],
                               "rows": key[3], "ranks": len(latencies),
                               "worst_rank": worst, "worst_rank_total_ms": latencies[worst]})
        return {"mode": "assumed_global_ids_unverified", "requests": joined,
                "total_ms_by_rows": percentiles([r["total_ms"] for r in records if r["total_ms"] is not None])}, dropped
    # Honest fallback: per-rank observations only; no cross-rank claim is made.
    ranked: dict[tuple, list[float]] = {}
    per_rank = []
    for record in records:
        if record["total_ms"] is None:
            continue
        per_rank.append({"source": record["source"], "execution_lane": record.get("execution_lane"),
                         "request_id": record.get("request_id"), "layer_id": record.get("layer_id"),
                         "rows": record.get("rows"), "total_ms": record["total_ms"]})
        ranked.setdefault((record["source"], record["rows"]), []).append(record["total_ms"])
    return {"mode": "per_rank_observation",
            "observations": per_rank,
            "by_rank_rows": [{"source": key[0], "rows": key[1], **percentiles(values)}
                             for key, values in sorted(ranked.items(), key=lambda item: str(item[0]))]}, 0


def summarize(paths: list[Path], tp: int, rows_filter: set[int] | None = None,
              start: str | None = None, end: str | None = None, run_id: str | None = None,
              assume_global_request_id: bool = False, include_raw: bool = True) -> dict:
    records = load(paths, tp, rows_filter, start, end)
    workers = [r for r in records if r["kind"] == "worker"]
    transport = [r for r in records if r["kind"] == "transport"]
    coordinator = [r for r in records if r["kind"] not in ("worker", "transport")]
    worker = aggregate_worker(workers)
    transport_summary, dropped = aggregate_transport(transport, run_id, assume_global_request_id)
    summary = {
        "tp": tp,
        "filters": {"rows": sorted(rows_filter) if rows_filter else None, "start": start, "end": end},
        "counts": {"workers": len(workers), "coordinator": len(coordinator), "transport": len(transport)},
        "worker_rank_groups": worker["rank_groups"],
        "worst_rank_median": worker["worst_rank_median"],
        "coordinator_phases": aggregate_coordinator(coordinator),
        "dspark": aggregate_dspark(coordinator),
        "transport": transport_summary,
        "matchability": {
            "worker_coordinator_join": "none: no request id; (layer, rows) buckets only",
            "worker_aggregate": "worst-rank median over (layer, rows), not a per-request critical path",
            "histogram_kind": "binned expert-row-count distribution (1..16 + 17+ tail), not expert ids",
            "transport_mode": transport_summary["mode"],
            "transport_cross_rank": ("requires --run-id plus a pre-verified global request id; "
                                     "an explicit unverified assumption, never a verified join"),
            "transport_unjoinable_records": dropped,
            "weight_footprint": "unique_expert_weight_bytes is a resident-footprint estimate, not traffic",
            "emitter_sources": [
                "v41_experts/execution.rs 'native expert execution'",
                "v41_backbone_lane.rs 'target experts[ with TP2 shared]' (routed_us/dispatch_us/shared_and_collect_us)",
                "v41_experts/coordinator.rs 'target collection' (shared_copy/upload/receive/reduce)",
                "v41_native_serve/scheduler.rs 'native scheduler round'",
                "v41_native_serve/scheduler/independent.rs 'native independent lane round'",
                "ds41rt-transport/src/verbs/local.rs 'protocol_v2_verbs_persistent_server_roundtrip_timing'",
            ],
        },
        "scope": ("Worker kernel/compact normalized by owned_rows*2304/tp (kernel-extent variant for padding). "
                  "Nested/overlapping coordinator phases are compared phase-by-phase, never summed. "
                  "dSpark acceptance uses accepted_inputs minus requests over proposed drafts; "
                  "tokens/round uses emitted only."),
    }
    if include_raw:
        summary["raw"] = {"workers": workers, "coordinator": coordinator, "transport": transport}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--tp", type=int, required=True, choices=[2, 3, 4])
    parser.add_argument("--rows", type=int, action="append", default=None,
                        help="keep only these request row counts (repeatable)")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--run-id")
    parser.add_argument("--assume-global-request-id", action="store_true")
    parser.add_argument("--no-raw", action="store_true", help="omit raw records from the output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.logs, args.tp, set(args.rows) if args.rows else None,
                        args.start, args.end, args.run_id, args.assume_global_request_id,
                        include_raw=not args.no_raw)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Parsed {len(args.logs)} log(s) tp={args.tp} rows={args.rows} -> {args.output}")
