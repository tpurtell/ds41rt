#!/usr/bin/env python3
"""Render the two v7 per-quant performance reports from the measured package.

Both reports are generated from the bench JSONs under
~/.cache/ds41rt-v7-package/performance so a table can never disagree with the
raw results. Unavailable or nonqualifying results render as an em dash;
nothing is estimated. Partial prefill matrices are explicitly provisional.
"""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

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
    ("counting", "Counting 1–200"),
]

QUANTS = {
    "nvfp4": dict(
        file="release-v7-nvfp4-performance.md",
        title="NVIDIA DeepSeek-V4.1-Flash-NVFP4 (W4A4)",
        checkpoint="nvidia/DeepSeek-V4.1-Flash-NVFP4",
        layouts=[("1x", "single", "1x RTX PRO 6000 + 4x Spark",
                  "4 full-width TP1 layers resident on the card; layers 4-39 on the Sparks."),
                 ("2x", "dual", "2x RTX PRO 6000 + 4x Spark",
                  "20 TP2 layers resident across the pair; layers 20-39 on the Sparks.")],
        note="Weights are ModelOpt NVFP4 (W4A4): E2M1 payload with E4M3 K16 block "
             "scales, activations quantized in-kernel from BF16 rows.",
    ),
    "exl3": dict(
        file="release-v7-exl3-k2-performance.md",
        title="diffbot DeepSeek-V4.1-Flash-EXL3 (2.0 bpw, K=2)",
        checkpoint="diffbot/DeepSeek-V4.1-Flash-EXL3-2.0bpw-2x-RTX-PRO-6000",
        layouts=[("1x", "single", "1x RTX PRO 6000 (32 GiB budget) + 2x Spark",
                  "Compact RTX 5090 memory simulation: total device ceiling includes 2 GiB runtime headroom; "
                  "layer 0 full-width on RTX and layers 1–39 as TP2 on exactly two Sparks; "
                  "prefill batches capped at 256 and global KV pool 2 GiB. "
                  "See [residency and qualification](release-v7-exl3-compact.md). "
                  "This is not a measurement on RTX 5090 hardware."),
                 ("2x", "dual", "2x RTX PRO 6000, no Spark",
                  "All 40 routed-expert layers resident as TP2 on the pair; no Spark worker.")],
        note="Uniform K=2 trellis with a two-tier kernel family; the paired 3.25 bpw "
             "publication uses the second tier.",
    ),
}

PENDING_NVFP4 = ["1x and 2x completed tool-call evaluation runs"]
PENDING_EXL3 = ["2x completed tool-call evaluation runs (1x compact carries three completed runs)",
                "Reasoning-code completion qualification: failed samples remain disclosed, not counted as quality passes",
                "RTX 5090 hardware performance (only same-capability grid checks on RTX PRO 6000, not physical RTX 5090 tests)"]
PENDING_COMMON = [
    "Fresh target-only decode; retained-context decode including the separate 2K control; "
    "counting/code/topic concurrency scaling and mixed-traffic sweeps: not qualified here to the v5/v6 scope",
    "Per-layout startup, memory and cache-capacity qualification; adaptive draft acceptance and "
    "fixed-history quant agreement: not replaced by historical official-image or v5 EXL3 results",
    "Per-campaign engine/SparkInfer revisions, quant snapshot and binary/launch identity, "
    "KV/PLE and TP2 controls, plus power/clock evidence (including stock-memory settings)",
    # Superseded: the images were built for both roles and published after the
    # review that added this line, so the report must not still claim otherwise.
    "v7 release images are built and published "
    "(ghcr.io/tpurtell/ds41rt-coordinator:v7 sha256:53af6703, "
    "ghcr.io/tpurtell/ds41rt-spark:v7 sha256:81918fe4); "
    "physical RTX 5090 validation remains owed",
]


# Where the tool evaluation writes its per-configuration runs, relative to the
# performance package. Names match the --output-dir used for each campaign.
TOOL_EVAL_DIRS = {
    ("nvfp4", "1x"): "nvfp4-single",
    ("nvfp4", "2x"): "nvfp4-dual",
    ("exl3", "1x"): "exl3-compact",
    ("exl3", "2x"): "exl3-dual",
}


def load_tool_eval(package: Path, quant: str, layout: str):
    """Completed evaluation runs for one configuration, or None.

    Each run carries its aggregate summary (points, status counts, output cap)
    and every scenario that did not fully pass. The run's own `tool-eval.json`
    is preferred for the detail because the aggregate's `failures` list holds
    only `status == "fail"`, which would hide partially credited scenarios;
    the aggregate list is the fallback.
    """
    name = TOOL_EVAL_DIRS.get((quant, layout))
    if not name:
        return None
    root = package.parent / "tool-eval" / name
    aggregate = root / "summaries.json"
    if not aggregate.is_file():
        return None
    try:
        summaries = json.loads(aggregate.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(summaries, list) or not summaries:
        return None
    runs = []
    for index, summary in enumerate(summaries, start=1):
        detail = None
        candidate = root / f"run-{index:02d}" / "tool-eval.json"
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text())
                detail = [result for result in
                          (payload.get("scores") or {}).get("scenario_results") or []
                          if result.get("status") != "pass"]
            except (json.JSONDecodeError, OSError, AttributeError):
                detail = None
        reasons = {failure.get("scenario_id"): failure.get("summary")
                   for failure in summary.get("failures") or []}
        if detail is None:
            detail = [dict(failure, status="fail") for failure in summary.get("failures") or []]
        else:
            # The per-run results carry status and points but not the reason;
            # take it from the aggregate so a failure is never listed bare.
            detail = [dict(result, summary=result.get("summary") or reasons.get(result.get("scenario_id")))
                      for result in detail]
        runs.append((summary, detail))
    return runs


def load_prefill(package: Path, stem: str):
    """The full matrix if it has cells, else the base-0 row, else whatever
    partial file exists. The best-prefill cell comes from the base-0 row in
    every configuration measured so far."""
    document = load(package, f"{stem}-prefill")
    if document is None:
        return None
    cells = document.get("cells") or []
    if any(c.get("median_effective_prefill_tokens_per_second") for c in cells):
        return document
    # A campaign that ran but produced no usable cell: the reduced base-0 row
    # is still a real measurement, so prefer it to reporting nothing. Files
    # that do not exist are not probed, which keeps reads to one per source.
    return load(package, f"{stem}-prefill-base0") or document


def load(package: Path, stem: str):
    path = package / f"{stem}.json"
    try:
        document = json.loads(path.read_text())
    except (ValueError, OSError):
        return None
    return document if isinstance(document, dict) else None


def decode_cells(document):
    if not document or not document.get("samples"):
        return None
    per_case = {}
    for sample in document["samples"]:
        rate = sample.get("observed_decode_tokens_per_second")
        if rate is not None:
            per_case.setdefault(sample.get("case"), []).append(rate)
    return (document.get("median_weighted_observed_decode_tokens_per_second"),
            {case: statistics.median(values) for case, values in per_case.items()})


def prefill_matrix(document):
    cells = (document or {}).get("cells") or []
    usable = [c for c in cells if c.get("median_effective_prefill_tokens_per_second")]
    if not usable:
        return None, None, None
    bases = sorted({c["base_context_tokens"] for c in usable})
    suffixes = sorted({c["suffix_tokens"] for c in usable})
    lookup = {(c["base_context_tokens"], c["suffix_tokens"]):
              c["median_effective_prefill_tokens_per_second"] for c in usable}
    return bases, suffixes, lookup


def prefill_complete(document):
    return bool(document and document.get("passed") is True and document.get("completed_ns"))


def best_prefill(document):
    if not prefill_complete(document):
        return None
    values = [c.get("median_effective_prefill_tokens_per_second")
              for c in (document or {}).get("cells") or []]
    values = [value for value in values if value]
    return max(values) if values else None


def fmt(value, thousands=False):
    if value is None:
        return "—"
    return f"{value:,.0f}" if thousands or value >= 1000 else f"{value:,.2f}"


def change(second, first):
    if second is None or first in (None, 0):
        return "—"
    return f"{100 * (second / first - 1):+.1f}%"


def render(quant: str, package: Path) -> str:
    spec = QUANTS[quant]
    documents = {layout: (load(package, f"{stem}-{quant}-dspark"),
                          load_prefill(package, f"{stem}-{quant}"))
                 for layout, stem, _, _ in spec["layouts"]}
    measured = {layout: (decode_cells(decode), prefill)
                for layout, (decode, prefill) in documents.items()}
    lines = [
        f"# DS41RT v7 performance - {spec['title']}",
        "",
        f"Checkpoint `{spec['checkpoint']}`. {spec['note']}",
        "",
        "The release protocol specifies **400 W per RTX card and standard memory speed, without a memory overclock**. "
        "Reported throughput cells use three samples. Recorded requests use temperature zero, "
        "high-effort thinking for reasoning code and thinking disabled for the other cases. "
        "Decode uses C1 dSpark; reasoning throughput includes reasoning and final-answer tokens.",
        "",
        "These are new v7 quant campaigns, not the historical official-image "
        "[v6 measurements](release-v6-performance.md) shown below the headline in the "
        "[README](../README.md#performance). V5 EXL3 results use a different checkpoint and are not substituted here.",
        "",
        "**Provenance.** Raw records are under `~/.cache/ds41rt-v7-package/performance/`; "
        "the filenames and SHA-256 digests below identify the selected inputs. They preserve "
        "request controls, timestamps, corpus/tokenizer hashes and individual samples, but their "
        "`model` is an API alias, not proof of the quant snapshot. They do not bind each campaign "
        "to engine/SparkInfer revisions, release-image hashes or power/clock telemetry. "
        "Those missing bindings remain owed; the protocol above is not a claim that every hardware control "
        "is independently verified by these JSON files. V7 release Docker images have not been built or published.",
        "",
        "Decode cells are medians of `observed_decode_tokens_per_second` by case; weighted decode "
        "is `median_weighted_observed_decode_tokens_per_second`: the median of repeat-level "
        "weighted token/time ratios, not an average of the case medians (nine categories; "
        "natural/schema JSON weight 0.5 each, other categories 1, counting excluded). "
        "Best prefill is the maximum `median_effective_prefill_tokens_per_second` over a completed, "
        "passing matrix. Change is `(2x / 1x - 1) × 100`, calculated before display rounding.",
        "Each report table is generated from the raw bench output in the v7 package by "
        "`scripts/render-ds41-v7-quant-reports.py`; an em dash means no usable qualifying result "
        "is available in the selected package, not necessarily that a measurement was never attempted. "
        "Best prefill requires a completed, passing campaign; incomplete matrices are provisional.",
        "",
    ]
    for _layout, stem, _label, _detail in spec["layouts"]:
        for kind in ("dspark", "prefill"):
            path = package / f"{stem}-{quant}-{kind}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "unavailable"
            lines.append(f"- `{path.name}` — SHA-256 `{digest}`")
    if quant == "exl3":
        lines += ["", "The [compact evidence manifest](release-v7-exl3-compact-evidence.json) "
                  "additionally records the checkpoint snapshot, WIP artifacts, placement and raw-source hashes. "
                  "Its `checkpoint_revision` identifies the checkpoint, not an engine commit. "
                  "The linked compact telemetry records a 400 W power limit, but does not establish "
                  "stock-clock settings or controls for the earlier dual/NVFP4 campaigns. "
                  "The [compact qualification record](release-v7-exl3-compact.md) distinguishes "
                  "same-capability grid checks on RTX PRO 6000 from untested physical RTX 5090 hardware."]
    lines += ["", "## Configurations", ""]
    for _layout, _stem, label, detail in spec["layouts"]:
        lines.append(f"- **{label}** — {detail}")
    if quant == "exl3":
        lines += ["", "EXL3 Change compares 2x RTX PRO 6000 with no Sparks against 1x RTX PRO 6000 "
                  "capped at 32 GiB plus 2x Spark, not isolated second-GPU scaling."]
    lines += ["", "## Headlines", "", "Tokens/s. Prefill is the best cell median; decode is C1 dSpark. "
              "Weighted decode excludes counting. Change compares the two configurations defined above.", ""]
    header = ["Measurement", "1 RTX", "2 RTX"]
    if len(spec["layouts"]) > 1:
        header.append("Change")
    lines += ["| " + " | ".join(header) + " |",
              "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    rows = [
        ("Best prefill", lambda d, p: best_prefill(p), True),
        ("Counting decode", lambda d, p: (d or ({}, {}))[1].get("counting"), False),
        ("Weighted decode", lambda d, p: (d or (None, {}))[0], False),
        ("C1 code decode", lambda d, p: (d or ({}, {}))[1].get("code"), False),
    ]
    for label, extract, whole in rows:
        values = [extract(measured[layout][0], measured[layout][1])
                  for layout, _s, _lb, _d in spec["layouts"]]
        row = [label] + [fmt(v, thousands=whole) for v in values]
        if len(values) > 1:
            row.append(change(values[1], values[0]))
        lines.append("| " + " | ".join(row) + " |")
    if any(s.get("passed") is False for decode, _prefill in documents.values()
           for s in (decode or {}).get("samples", [])):
        lines += ["", "**Completion failures remain included in these rates.** "
                  "See [Quality and evaluation](#quality-and-evaluation) for the exact failed samples "
                  "and output caps; this is not a fully passing quality campaign."]
    checks = ["## Quality and evaluation", "",
              "**Decode completion checks.** These check response completion and limited objective structure, "
              "not general prose quality, tool calling or fixed-history quant agreement. "
              "Failed samples remain in throughput and in the denominators.", ""]
    for layout, _stem, label, _detail in spec["layouts"]:
        document = documents[layout][0]
        if not document or not document.get("samples"):
            continue
        samples = document["samples"]
        failed = [s for s in samples if s.get("passed") is False]
        passed = sum(s.get("passed") is True for s in samples)
        unknown = len(samples) - passed - len(failed)
        checks.append(f"- **{label}**: {passed}/{len(samples)} sample checks explicitly passed; "
                     f"{len(failed)} failed; {unknown} unknown/unreported. "
                     "These completion checks do not establish a full benchmark pass.")
        for sample in failed:
            checks.append(f"  - {sample.get('case', 'unknown')}, repeat {sample.get('repeat', 'unknown')}: "
                          f"finish reason `{sample.get('finish_reason', 'error')}`; "
                          f"final response {'nonempty' if (sample.get('content') or '').strip() else 'empty'}. "
                          f"Output tokens: {sample.get('usage', {}).get('completion_tokens', 'unreported')}; "
                          f"request cap: {sample.get('request', {}).get('max_tokens', 'unreported')}. "
                          "Throughput above includes this sample; it is not a successful quality result.")
    # Tool-call evaluation: a published result requires a completed run, and
    # every failed scenario is listed rather than folded into a pass rate.
    evaluated = []
    for layout, _stem, label, _detail in spec["layouts"]:
        runs = load_tool_eval(package, quant, layout)
        if not runs:
            continue
        for index, (summary, detail) in enumerate(runs, start=1):
            evaluated.append(
                f"- **{label}, run {index}**: {summary.get('total_points')}/{summary.get('total_max')}"
                f" points (basic {summary.get('basic_points')}/{summary.get('basic_max')},"
                f" hard {summary.get('hard_points')}/{summary.get('hard_max')});"
                f" statuses {summary.get('statuses')}; output cap {summary.get('output_cap')}"
                f" ({summary.get('output_cap_source')}).")
            for result in detail:
                note = (result.get("summary") or result.get("error") or "").strip()
                note = " ".join(note.split())[:180]
                evaluated.append(
                    f"  - `{result.get('scenario_id')}` {result.get('status', 'non-passing')}"
                    f" ({result.get('points')} points): {note}")
    if evaluated:
        checks += ["", "**Tool-call evaluation.** Completed runs; every scenario that did not fully pass is listed."] + evaluated
    else:
        checks += ["", "Tool-call evaluation, adaptive draft acceptance and fixed-history quant agreement "
                   "have no completed results published here. Historical official or v5 EXL3 quality scores "
                   "are not evidence for these new quants."]
    lines += ["", "## Content-type decode", "", "Median C1 dSpark tokens/s; three samples per case. "
              "Schema JSON is grammar-constrained. See Quality and evaluation below for failed completion checks.", ""]
    header = ["Case", "1 RTX dSpark", "2 RTX dSpark"]
    if len(header) > 1:
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        for case, name in CASES:
            row = [name]
            for layout, _s, _lb, _d in spec["layouts"]:
                row.append(fmt((measured[layout][0] or (None, {}))[1].get(case)))
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Prefill", "",
              "Median effective tokens/s: uncached suffix tokens divided by client time to first content, "
              "not isolated GPU prefill time. Completed matrices contain 30 cells, each with one excluded "
              "shape warmup and three measured samples with verified parent reuse. K = 1,024 tokens.", ""]
    for layout, _stem, label, _detail in spec["layouts"]:
        bases, suffixes, lookup = prefill_matrix(measured[layout][1])
        lines.append(f"**{label}**")
        lines.append("")
        if not prefill_complete(measured[layout][1]):
            lines += ["_Completed, passing prefill status has not been established. "
                      "Any available partial measurements below are provisional and excluded from Best prefill._", ""]
        if not bases:
            lines += ["_No finalized prefill cell summaries are available in the selected package. "
                      "Raw partial samples, if present, are not a completed matrix or a qualifying best._", ""]
            continue
        lines += ["| Retained base | " + " | ".join(f"+{s // 1024}K" for s in suffixes) + " |",
                  "|" + "|".join(["---"] + ["---:"] * len(suffixes)) + "|"]
        for base in bases:
            row = [f"{base // 1024}K"]
            row += [fmt(lookup.get((base, suffix)), thousands=True) for suffix in suffixes]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += checks + [""]
    if quant == "nvfp4":
        lines += ["The 1x prefill campaign was interrupted by a service restart "
                  "([campaign record](release-v7-plan.md)). Its raw file retains partial samples, "
                  "but no finalized cell summaries or completed, passing matrix; no best is estimated.", ""]
    pending = list(PENDING_NVFP4 if quant == "nvfp4" else PENDING_EXL3) + PENDING_COMMON
    for layout, _stem, label, _detail in spec["layouts"]:
        if not prefill_complete(measured[layout][1]):
            pending.append(f"{label} full prefill: no completed, passing campaign result available; "
                           "any partial results are provisional")
    lines += ["## Outstanding measurements and qualification", ""]
    lines += [f"- {item}" for item in pending]
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path,
                        default=Path.home() / ".cache/ds41rt-v7-package/performance")
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    for quant, spec in QUANTS.items():
        path = args.output_dir / spec["file"]
        path.write_text(render(quant, args.package))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
