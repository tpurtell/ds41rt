#!/usr/bin/env python3
"""Render the DS41RT v9 official/TP6 performance report from a manifest.

Every table cell is derived from a raw bench JSON named by the manifest. A
missing file, a missing field or an arm that was not measured renders as an em
dash; nothing is estimated, interpolated or carried over from another

campaign. Negative results and failed checks are rendered from the manifest's
`failures` list rather than dropped.

The headline tables deliberately carry no delta column: arms are reported side
by side with their own identity, and any change claim belongs in a separate
review that also states the repeat spread it was compared against.

Usage:
  scripts/render-ds41-v9-tp6-reports.py --manifest MANIFEST.json \
      --package DIR --output FILE.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

DASH = "\u2014"
CASES = [
    ("code", "Code"),
    ("code-reasoning", "Code with reasoning"),
    ("math", "Math"),
    ("fable", "Fable"),
    ("hello", "Hello"),
    ("topic", "Topic"),
    ("structured-json", "Natural JSON"),
    ("structured-json-schema", "Schema JSON"),
    ("multilingual", "Multilingual"),
    ("counting", "Counting 1-200"),
]


class Missing(Exception):
    """A manifest or raw-data problem that must stop the render."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, thousands: bool = False):
    if value is None:
        return DASH
    if isinstance(value, float):
        return f"{value:,.0f}" if thousands else f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}" if thousands else str(value)
    return str(value)


def load_json(package: Path, relative):
    """Return (document, sha256) or (None, None) when absent."""
    if relative in (None, ""):
        return None, None
    path = (package / relative) if not Path(relative).is_absolute() else Path(relative)
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text()), sha256(path)
    except (OSError, json.JSONDecodeError):
        return None, None


def decode_summary(doc):
    if not doc:
        return None
    by_case = {}
    for sample in doc.get("samples", []):
        value = sample.get("observed_decode_tokens_per_second")
        if value is not None:
            by_case.setdefault(sample.get("case"), []).append(value)
    cases = {case: statistics.median(values) for case, values in by_case.items() if values}
    repeats = [
        r.get("weighted_observed_decode_tokens_per_second")
        for r in doc.get("repeat_summaries", [])
    ]
    repeats = [r for r in repeats if r is not None]
    spread = None
    if len(repeats) > 1 and statistics.median(repeats):
        spread = (max(repeats) - min(repeats)) / statistics.median(repeats) * 100.0
    assessed = [s for s in doc.get("samples", []) if s.get("objective_checks_passed") is not None]
    return {
        "cases": cases,
        "weighted": doc.get("median_weighted_observed_decode_tokens_per_second"),
        "weighted_repeats": repeats,
        "weighted_spread_percent": spread,
        "samples": len(doc.get("samples", [])),
        "serving_completed": sum(bool(s.get("serving_completed")) for s in doc.get("samples", [])),
        "objective_assessed": len(assessed),
        "objective_passed": sum(bool(s.get("objective_checks_passed")) for s in assessed),
        "passed": doc.get("passed"),
        "corpus_sha256": doc.get("corpus_sha256"),
        "tokenizer_sha256": doc.get("tokenizer_sha256"),
        "repeats": doc.get("repeats"),
        "nonce_seed": doc.get("nonce_seed"),
    }


def prefill_matrix(doc):
    if not doc:
        return None
    cells = doc.get("cells")
    if not cells:
        return None
    bases = doc.get("bases")
    suffixes = doc.get("suffixes")
    if not bases or not suffixes:
        return None
    lookup = {
        (c.get("base_context_tokens"), c.get("suffix_tokens")):
        c.get("median_effective_prefill_tokens_per_second")
        for c in cells
    }
    grid = [[lookup.get((b, s)) for s in suffixes] for b in bases]
    values = [v for row in grid for v in row if v is not None]
    return {
        "bases": bases,
        "suffixes": suffixes,
        "grid": grid,
        "best": max(values) if values else None,
        "complete": len(values) == len(bases) * len(suffixes),
        "cells": len(values),
        "expected_cells": len(bases) * len(suffixes),
    }


def retained_summary(doc):
    if not doc:
        return None
    rows = []
    for summary in doc.get("context_summaries", []):
        rows.append(
            (
                summary.get("context_tokens"),
                summary.get("weighted_observed_decode_tokens_per_second"),
            )
        )
    return rows or None


def concurrency_summary(doc):
    if not doc:
        return None
    rows = [
        (s.get("concurrency"), s.get("median_aggregate_tps"))
        for s in doc.get("summaries", [])
    ]
    return [r for r in rows if r[0] is not None] or None


def kernel_tiling(doc):
    """FFN microbenchmark cells. The kernel package owns its schema; we render
    only the documented keys and treat anything else as unavailable."""
    if not doc:
        return None
    cells = doc.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    out = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        out.append(
            {
                "topology": cell.get("topology"),
                "workload": cell.get("workload"),
                "width": cell.get("width"),
                "capacity": cell.get("capacity"),
                "rows": cell.get("rows"),
                "amortised_us": cell.get("amortised_us", cell.get("amortised")),
                "cold_us": cell.get("cold_us", cell.get("cold")),
            }
        )
    return out or None


def require_provenance(manifest: dict):
    problems = []
    for key in ("release", "checkpoint", "images", "network", "thresholds", "arms"):
        if key not in manifest:
            problems.append(f"manifest missing '{key}'")
    for image_key in ("coordinator", "spark_expert"):
        image = manifest.get("images", {}).get(image_key, {})
        for field in ("tag", "digest", "revision"):
            if not image.get(field):
                problems.append(f"images.{image_key}.{field} is required")
    for arm in manifest.get("arms", []):
        for field in ("id", "quant", "topology", "rtx", "sparks", "config_sha256", "raw"):
            if field not in arm:
                problems.append(f"arm {arm.get('id', '?')} missing '{field}'")
    if problems:
        raise Missing("; ".join(problems))


def render(manifest: dict, package: Path) -> str:
    require_provenance(manifest)
    arms = manifest["arms"]
    loaded = []
    for arm in arms:
        raw = arm.get("raw", {})
        docs = {name: load_json(package, raw.get(name)) for name in raw}
        loaded.append(
            {
                "arm": arm,
                "decode": decode_summary(docs.get("decode", (None, None))[0]),
                "prefill": prefill_matrix(docs.get("prefill", (None, None))[0]),
                "retained": retained_summary(docs.get("retained", (None, None))[0]),
                "concurrency": concurrency_summary(
                    docs.get("concurrency_counting", (None, None))[0]
                ),
                "tiling": kernel_tiling(docs.get("kernel_tiling", (None, None))[0]),
                "digests": {
                    name: (raw.get(name), digest)
                    for name, (_, digest) in docs.items()
                    if digest
                },
                "present": {name: docs[name][0] is not None for name in docs},
            }
        )

    ck = manifest["checkpoint"]
    images = manifest["images"]
    network = manifest["network"]
    thresholds = manifest["thresholds"]
    lines = []
    add = lines.append

    add(f"# DS41RT {manifest['release']} performance: official checkpoint and TP6")
    add("")
    add(
        "Generated by `scripts/render-ds41-v9-tp6-reports.py` from the raw bench "
        "outputs named below. Every cell is derived from a recorded sample; an em "
        f"dash ({DASH}) means no qualifying measurement is present in this package, "
        "not that a run was never attempted. Nothing is estimated, interpolated or "
        "substituted from another campaign."
    )
    add("")
    add("**Provenance.**")
    add("")
    add(f"- Model quant: `{ck['model_id']}` @ `{ck['revision']}` ({ck['quant']})")
    add(f"- Coordinator image: `{images['coordinator']['tag']}` `{images['coordinator']['digest']}` (revision `{images['coordinator']['revision']}`)")
    add(f"- Spark expert image: `{images['spark_expert']['tag']}` `{images['spark_expert']['digest']}` (revision `{images['spark_expert']['revision']}`)")
    add(
        f"- Network: Spark rails at **{network['spark_link_mbps']:,} Mb/s** negotiated, "
        f"topology `{network['rails']}`; evidence `{network['evidence']}`"
    )
    if network.get("dual_lane_note"):
        add(f"- Dual-lane note: {network['dual_lane_note']}")
    add(
        f"- Change thresholds (median-of-3 same-image runs): <{thresholds['noise_percent']}% "
        f"noise, {thresholds['noise_percent']}-{thresholds['weak_percent']}% weak, "
        f">{thresholds['credible_percent']}% credible. Basis: {thresholds['basis']}"
    )
    add("")

    # ---- Headlines (no delta column by design) ----
    add("## Headlines")
    add("")
    add("Best prefill is the maximum passing matrix cell; decode is C1 dSpark.")
    add("")
    header = "| Measurement | " + " | ".join(a["arm"]["id"] for a in loaded) + " |"
    add(header)
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    rows = [
        (
            "Best prefill",
            [a["prefill"]["best"] if a["prefill"] else None for a in loaded],
            True,
        ),
        (
            "Counting decode",
            [a["decode"]["cases"].get("counting") if a["decode"] else None for a in loaded],
            False,
        ),
        (
            "Weighted decode",
            [a["decode"]["weighted"] if a["decode"] else None for a in loaded],
            False,
        ),
        (
            "C1 code decode",
            [a["decode"]["cases"].get("code") if a["decode"] else None for a in loaded],
            False,
        ),
    ]
    for label, values, thousands in rows:
        add(f"| {label} | " + " | ".join(fmt(v, thousands) for v in values) + " |")
    add("")

    # ---- Arms ----
    add("## Configurations")
    add("")
    for a in loaded:
        arm = a["arm"]
        add(f"- **{arm['id']}** - {arm['quant']} `{arm['topology']}`, "
            f"{arm['rtx']} RTX + {arm['sparks']} Spark. "
            f"config sha256 `{arm['config_sha256']}`.")
        if arm.get("runtime_flags"):
            flags = ", ".join(f"{k}={v}" for k, v in sorted(arm["runtime_flags"].items()))
            add(f"  - resolved: {flags}")
        if arm.get("warmup") or arm.get("repeats"):
            add(f"  - warmup: {arm.get('warmup') or DASH}; repeats: {fmt(arm.get('repeats'))}")
        if arm.get("comparison_group"):
            add(f"  - comparison group: `{arm['comparison_group']}` (arms in one group may "
                "differ in Spark count / TP world by design)")
        if arm.get("images"):
            override = arm["images"]
            add(f"  - image override: coordinator `{override['coordinator']['tag']}` "
                f"`{override['coordinator']['digest']}`, spark `{override['spark_expert']['tag']}` "
                f"`{override['spark_expert']['digest']}`")
    add("")

    # ---- Weight and residency ----
    add("## Weight and residency")
    add("")
    add(
        "Deployments differ in per-rank resident weights and remote layer count; a "
        "4-Spark vs 6-Spark comparison reports these so the throughput difference is "
        "read together with the configuration difference."
    )
    add("")
    add("| Arm | Topology | RTX | Sparks | Resident bytes/rank | Remote dispatched layers | Worker allocated layers |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for a in loaded:
        arm = a["arm"]
        add(
            f"| {arm['id']} | {fmt(arm.get('topology'))} | {fmt(arm.get('rtx'))} | "
            f"{fmt(arm.get('sparks'))} | {fmt(arm.get('resident_bytes_per_rank'), True)} | "
            f"{fmt(arm.get('remote_layers'))} | {fmt(arm.get('worker_allocated_layers'))} |"
        )
    add("")
    for a in loaded:
        if a["arm"].get("weight_note"):
            add(f"- **{a['arm']['id']}**: {a['arm']['weight_note']}")
    add("")

    # ---- Cost policy ----
    add("## Speculative cost policy")
    add("")
    add(
        "A `profile=None` input path does not prove the resolved mode. Each arm "
        "records the resolved mode under its normal defaults and, for attribution, "
        "the controlled legacy side arm."
    )
    add("")
    add("| Arm | Default-mode resolved | Legacy side-arm resolved | DS41RT_ADAPTIVE_COST_MODE | Evidence |")
    add("|---|---|---|---|---|")
    for a in loaded:
        policy = a["arm"].get("cost_policy")
        if policy:
            add(
                f"| {a['arm']['id']} | {fmt(policy.get('default_mode'))} | "
                f"{fmt(policy.get('legacy_mode'))} | {fmt(policy.get('adaptive_mode_env'))} | "
                f"{fmt(policy.get('evidence'))} |"
            )
        else:
            add(f"| {a['arm']['id']} | {DASH} | {DASH} | {DASH} | {DASH} |")
    add("")

    # ---- Content-type decode ----
    add("## Content-type decode")
    add("")
    add("Median C1 dSpark tokens/s from the recorded samples.")
    add("")
    add("| Case | " + " | ".join(a["arm"]["id"] for a in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    for case, label in CASES:
        values = [a["decode"]["cases"].get(case) if a["decode"] else None for a in loaded]
        add(f"| {label} | " + " | ".join(fmt(v) for v in values) + " |")
    add("")

    # ---- Repeat variance ----
    add("## Within-arm repeat variance")
    add("")
    add("Weighted-decode repeat values, so a reader can see the spread behind each median.")
    add("")
    add("| Arm | Repeats | Spread |")
    add("|---|---:|---:|")
    for a in loaded:
        if a["decode"]:
            reps = ", ".join(fmt(v) for v in a["decode"]["weighted_repeats"]) or DASH
            spread = a["decode"]["weighted_spread_percent"]
            add(f"| {a['arm']['id']} | {reps} | {DASH if spread is None else f'{spread:.1f}%'} |")
        else:
            add(f"| {a['arm']['id']} | {DASH} | {DASH} |")
    add("")

    # ---- Prefill ----
    add("## Prefill matrices")
    add("")
    add("Median effective tokens/s per retained-base/suffix cell. Incomplete matrices are provisional.")
    add("")
    for a in loaded:
        add(f"### {a['arm']['id']}")
        add("")
        pf = a["prefill"]
        if not pf:
            add(f"{DASH} no prefill package present.")
            add("")
            continue
        add(
            f"{pf['cells']}/{pf['expected_cells']} cells present"
            + ("" if pf["complete"] else " - **provisional, incomplete**")
        )
        add("")
        add("| Retained base | " + " | ".join(f"+{s//1024}K" for s in pf["suffixes"]) + " |")
        add("|---|" + "|".join("---:" for _ in pf["suffixes"]) + "|")
        for base, row in zip(pf["bases"], pf["grid"]):
            add(f"| {base//1024}K | " + " | ".join(fmt(v, True) for v in row) + " |")
        add("")

    # ---- Retained context ----
    add("## Decode over retained context")
    add("")
    add("Weighted nine-category dSpark tokens/s with verified prefix reuse.")
    add("")
    add("| Retained base | " + " | ".join(a["arm"]["id"] for a in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    contexts = sorted({c for a in loaded if a["retained"] for c, _ in a["retained"]})
    if contexts:
        for ctx in contexts:
            cells = []
            for a in loaded:
                value = None
                if a["retained"]:
                    value = next((v for c, v in a["retained"] if c == ctx), None)
                cells.append(fmt(value))
            add(f"| {ctx//1024}K | " + " | ".join(cells) + " |")
    else:
        add(f"| {DASH} | " + " | ".join(DASH for _ in loaded) + " |")
    add("")

    # ---- Concurrency ----
    add("## Concurrency scaling")
    add("")
    add("Median aggregate counting tokens/s from earliest first output to final completion.")
    add("")
    add("| Concurrency | " + " | ".join(a["arm"]["id"] for a in loaded) + " |")
    add("|---|" + "|".join("---:" for _ in loaded) + "|")
    concs = sorted({c for a in loaded if a["concurrency"] for c, _ in a["concurrency"]})
    if concs:
        for c in concs:
            cells = []
            for a in loaded:
                value = None
                if a["concurrency"]:
                    value = next((v for cc, v in a["concurrency"] if cc == c), None)
                cells.append(fmt(value))
            add(f"| C{c} | " + " | ".join(cells) + " |")
    else:
        add(f"| {DASH} | " + " | ".join(DASH for _ in loaded) + " |")
    add("")

    # ---- Kernel / FFN tiling ----
    add("## FFN kernel tiling")
    add("")
    add(
        "Component microbenchmark; this is **not** serving throughput. Rows/capacity are "
        "the actual measured dimensions and cover both decode-sized and prefill-sized batches."
    )
    add("")
    if any(a["tiling"] for a in loaded):
        add("| Arm | Topology | Workload | Width | Capacity | Rows | Amortised us | Cold us |")
        add("|---|---|---|---:|---:|---:|---:|---:|")
        for a in loaded:
            for cell in a["tiling"] or []:
                add(
                    f"| {a['arm']['id']} | {fmt(cell['topology'])} | {fmt(cell['workload'])} | "
                    f"{fmt(cell['width'])} | {fmt(cell['capacity'])} | {fmt(cell['rows'])} | "
                    f"{fmt(cell['amortised_us'])} | {fmt(cell['cold_us'])} |"
                )
    else:
        add(f"{DASH} no kernel tiling package present.")
    add("")

    # ---- Quality and retained failures ----
    add("## Quality, checks and retained failures")
    add("")
    add(
        "`Objective passed` is out of `Objective assessed`. Samples without an "
        "objective check are **unassessed, not passes**; a completed sample is not "
        "a passed sample."
    )
    add("")
    add("| Arm | Timed samples | Serving completed | Objective passed | Objective assessed | Harness passed |")
    add("|---|---:|---:|---:|---:|---|")
    for a in loaded:
        if a["decode"]:
            d = a["decode"]
            passed = d["passed"]
            add(
                f"| {a['arm']['id']} | {d['samples']} | {d['serving_completed']} | "
                f"{d['objective_passed']} | {d['objective_assessed']} | "
                f"{DASH if passed is None else ('yes' if passed else 'no')} |"
            )
        else:
            add(f"| {a['arm']['id']} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} |")
    add("")
    any_failure = False
    for a in loaded:
        for failure in a["arm"].get("failures", []) or []:
            any_failure = True
            scope = f" ({failure['scope']})" if failure.get("scope") else ""
            evidence = f" Evidence: `{failure['evidence']}`." if failure.get("evidence") else ""
            add(f"- **{a['arm']['id']}** - {failure['what']}{scope}: {failure['observed']}.{evidence}")
    if not any_failure:
        add("- No negative results are recorded in this manifest. Absence here is not a pass.")
    add("")

    # ---- Missing-data index ----
    add("## Missing-data index")
    add("")
    add("Every arm/artifact that rendered as an em dash, so nothing is omitted silently.")
    add("")
    add("| Arm | Artifact | State |")
    add("|---|---|---|")
    empty = True
    for a in loaded:
        for name, present in sorted(a["present"].items()):
            if not present:
                empty = False
                add(f"| {a['arm']['id']} | {name} | {DASH} not present |")
    if empty:
        add(f"| {DASH} | {DASH} | every declared artifact was present |")
    add("")

    # ---- Raw digests ----
    add("## Raw record digests")
    add("")
    add("| Arm | Artifact | SHA-256 |")
    add("|---|---|---|")
    for a in loaded:
        for name, (path, digest) in sorted(a["digests"].items()):
            add(f"| {a['arm']['id']} | {name} (`{path}`) | `{digest}` |")
    add("")
    return "\n".join(lines) + "\n"


def check(manifest: dict, package: Path) -> list:
    """Validate provenance and report declared artifacts that are absent."""
    require_provenance(manifest)
    problems = []
    for arm in manifest["arms"]:
        for name, relative in sorted((arm.get("raw") or {}).items()):
            if relative in (None, ""):
                problems.append(f"{arm['id']}: {name} not declared (renders as em dash)")
                continue
            path = (package / relative) if not Path(relative).is_absolute() else Path(relative)
            if not path.is_file():
                problems.append(f"{arm['id']}: {name} missing at {relative}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=None,
                        help="Directory that relative raw paths resolve against "
                             "(default: the manifest's directory)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="Validate provenance and artifact presence without "
                             "writing a report; exit 2 on a provenance error.")
    parser.add_argument("--strict", action="store_true",
                        help="With --check, also exit 2 when a declared artifact "
                             "is absent (default: report but do not fail).")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    package = args.package or args.manifest.parent
    if args.check:
        try:
            problems = check(manifest, package)
        except Missing as error:
            print(f"render-ds41-v9-tp6-reports: {error}", file=sys.stderr)
            return 2
        for problem in problems:
            print(problem)
        print(f"checked {len(manifest['arms'])} arm(s): {len(problems)} problem(s)")
        return 2 if (problems and args.strict) else 0
    if args.output is None:
        parser.error("--output is required unless --check is given")
    try:
        text = render(manifest, package)
    except Missing as error:
        print(f"render-ds41-v9-tp6-reports: {error}", file=sys.stderr)
        return 2
    args.output.write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
