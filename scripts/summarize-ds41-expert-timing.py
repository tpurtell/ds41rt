#!/usr/bin/env python3
"""Summarize Spark GPU/staging, RoCE boundaries and coordinator stage timings."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path


def summarize(paths):
    groups = {}
    for path in paths:
        for line in path.read_text().splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if "native expert execution" in line:
                kind = "gpu_and_staging"
            elif "protocol_v2_verbs_persistent_server_roundtrip_timing " in line:
                kind = "roce_server_boundary"
            elif "protocol_v2_expert_server_roundtrip_timing request_id=" in line:
                kind = "server_boundary"
            else:
                stage = re.search(r"\btarget (attention stages|query preparation|layer preparation|collection|experts|layer|step) ", line)
                if not stage:
                    continue
                kind = "coordinator_" + stage[1].replace(" ", "_")
            fields = {k: float(v) for k, v in re.findall(
                r"\b(\w+)=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", line)}
            histogram = re.search(r"expert_rows_histogram=\[([0-9, ]+)\]", line)
            if histogram:
                counts = [int(v) for v in histogram[1].split(",")]
                if "rows" not in fields or "active_experts" not in fields:
                    raise ValueError(f"expert row histogram without rows/active_experts in {path}")
                rows_value = fields["rows"]
                active_value = fields["active_experts"]
                # The line parser accepts any non-negative decimal; the histogram
                # contract only makes sense for integral counts, so reject
                # fractional or non-finite values instead of coercing them.
                if (not math.isfinite(rows_value) or rows_value < 1
                        or rows_value != int(rows_value)):
                    raise ValueError(f"non-integral expert row histogram rows in {path}")
                if (not math.isfinite(active_value) or active_value < 0
                        or active_value != int(active_value)):
                    raise ValueError(f"non-integral expert row histogram active_experts in {path}")
                rows_int = int(rows_value)
                active_int = int(active_value)
                tail_value = fields.get("expert_rows_tail_routes")
                if (tail_value is None or not math.isfinite(tail_value) or tail_value < 0
                        or tail_value != int(tail_value)):
                    raise ValueError(
                        f"non-integral expert row histogram expert_rows_tail_routes in {path}")
                tail_routes = int(tail_value)
                # A replicated group owns only a subset of the canonical six
                # routes per row: routes owned by other groups are masked to the
                # 384 sentinel and are not counted in this histogram. The owned
                # total is therefore inferred from the histogram and bounded by
                # rows * 6 instead of being assumed equal to it.
                owned_routes = sum((i+1)*v for i,v in enumerate(counts[:16])) + tail_routes
                if (len(counts) != 17 or sum(counts) != active_int
                        or any(v and i+1 > rows_int for i,v in enumerate(counts))
                        or not 17*counts[-1] <= tail_routes <= rows_int*counts[-1]
                        or owned_routes > rows_int*6
                        or (owned_routes == 0) != (active_int == 0)):
                    raise ValueError(f"inconsistent expert row histogram in {path}")
                # Optional future producer field: when an explicit owned-route
                # count is present it must agree with the histogram exactly, but
                # its absence is not an error.
                if "owned_routes" in fields:
                    declared = fields["owned_routes"]
                    if (not math.isfinite(declared) or declared != int(declared)
                            or int(declared) != owned_routes):
                        raise ValueError(
                            f"declared owned_routes disagrees with the histogram in {path}")
                fields["_expert_rows_histogram"] = counts
                fields["_owned_routes"] = owned_routes
                fields["_total_request_routes"] = rows_int*6
            rows = int(fields["rows"])
            key = (str(path), kind, rows)
            groups.setdefault(key, []).append(fields)
    if not groups:
        raise ValueError("no native expert or server-boundary timing records found")
    output = []
    for (source, kind, rows), records in sorted(groups.items()):
        metrics = {}
        for name in sorted(set.intersection(*(set(r) for r in records))):
            if not (name.endswith(("_us", "_ms", "_bytes"))
                    or name in {"active_experts", "max_expert_rows"}):
                continue
            values = sorted(r[name] for r in records)
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"invalid metric {name} in {source}")
            metrics[name] = {"mean": statistics.fmean(values),
                             "median": statistics.median(values),
                             "p95": values[math.ceil(len(values) * 0.95) - 1],
                             "max": values[-1]}
        group = {"source": source, "kind": kind, "rows": rows,
                 "samples": len(records), "metrics": metrics}
        histograms = [r for r in records if "_expert_rows_histogram" in r]
        if histograms:
            counts = [sum(r["_expert_rows_histogram"][i] for r in histograms) for i in range(17)]
            total_experts = sum(counts)
            owned_routes = sum(r["_owned_routes"] for r in histograms)
            total_request_routes = sum(r["_total_request_routes"] for r in histograms)
            distribution = []
            for i, count in enumerate(counts):
                routes = ((i+1)*count if i < 16 else
                          sum(r["expert_rows_tail_routes"] for r in histograms))
                distribution.append({"expert_rows": i+1 if i < 16 else "17+",
                                     "experts": count, "routes": int(routes),
                                     "expert_fraction": (count/total_experts) if total_experts else None,
                                     "route_fraction": (routes/owned_routes) if owned_routes else None})
            group["expert_row_distribution"] = distribution
            group["expert_row_distribution_samples"] = len(histograms)
            group["owned_routes"] = int(owned_routes)
            group["total_request_routes"] = int(total_request_routes)
            group["route_fraction_scope"] = "owned routes inferred from histogram"
        output.append(group)
    return {"scope": "Instrumented development workload. GPU event intervals separate expert execution and compaction; host upload/download exclude network transfer. Unique-expert packed bytes exclude repeated reads and are not measured DRAM traffic. Server callback includes staging and, for queued workers, queueing. Coordinator expert phase includes routing, shared FFN and response collection; shared FFN overlaps the dispatched remote request. Receive time includes waiting for remote compute and client handling. Coordinator upload_us measures host copy-call duration (staging/enqueue for async uploads), and reduce_us includes the final stream drain. Dispatch measures enqueue, not NIC send completion. No interval isolates pure link latency. Component medians are not additive; nested coordinator stage groups must not be summed together. Replicated expert groups own only a subset of the canonical six routes per row, so expert-row route fractions use histogram-inferred owned routes and owned_routes can be below total_request_routes; expert_fraction and route_fraction are null when their denominator is zero (an empty group) instead of dropping the distribution metadata.",
            "groups": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.logs)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(result['groups'])} timing groups.")
