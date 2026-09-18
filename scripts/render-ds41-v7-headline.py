#!/usr/bin/env python3
"""Render v7 headline cells from raw records, preserving historical v6 provenance."""
import argparse
import json
import statistics
from pathlib import Path

METRICS = ["Prefill", "Counting decode", "Weighted decode", "C1 code decode"]

# (column, quant, layout, provenance, values)
CONFIGS = [
    ("Official 1x", "official", "1x", "v6", {"Prefill": 7823.90, "Counting decode": 161.58,
                                             "Weighted decode": 92.00, "C1 code decode": 130.41}),
    ("Official 2x", "official", "2x", "v6", {"Prefill": 8355.22, "Counting decode": 221.64,
                                             "Weighted decode": 109.44, "C1 code decode": 155.70}),
] + [(f"{label} {layout}", quant, layout, "v7", dict.fromkeys(METRICS))
     for quant, label in (("nvfp4", "NVFP4"), ("exl3", "EXL3"))
     for layout in ("1x", "2x")]


def number(value):
    return "—" if value is None else f"{value:,.0f}" if value >= 1000 else f"{value:,.2f}"


def change(second, first):
    if second is None or first in (None, 0):
        return "—"
    return f"{100 * (second / first - 1):+.1f}%"


def load(package, stem):
    try:
        document = json.loads((package / f"{stem}.json").read_text())
    except (ValueError, OSError):
        return None
    return document if isinstance(document, dict) else None


def prefill_complete(document):
    return bool(document and document.get("passed") is True and document.get("completed_ns"))


def load_measurements(package):
    """Read each record once and build fresh cells, including on repeated calls."""
    configs = []
    documents = {}
    for column, quant, layout, provenance, defaults in CONFIGS:
        values = defaults.copy() if provenance == "v6" else dict.fromkeys(METRICS)
        configs.append((column, quant, layout, provenance, values))
        if provenance != "v7":
            continue
        stem = f"{'single' if layout == '1x' else 'dual'}-{quant}"
        decode = load(package, f"{stem}-dspark")
        prefill = load(package, f"{stem}-prefill")
        documents[column] = (decode, prefill)
        if decode and decode.get("samples"):
            per_case = {}
            for sample in decode["samples"]:
                rate = sample.get("observed_decode_tokens_per_second")
                if rate is not None:
                    per_case.setdefault(sample.get("case"), []).append(rate)
            for case, metric in (("code", "C1 code decode"), ("counting", "Counting decode")):
                if per_case.get(case):
                    values[metric] = statistics.median(per_case[case])
            values["Weighted decode"] = decode.get("median_weighted_observed_decode_tokens_per_second")
        if prefill_complete(prefill):
            rates = [cell.get("median_effective_prefill_tokens_per_second")
                     for cell in prefill.get("cells") or []]
            rates = [rate for rate in rates if rate is not None]
            values["Prefill"] = max(rates) if rates else None
    return configs, documents


def pairs(configs=None):
    by_quant = {}
    for _column, quant, layout, _prov, values in (CONFIGS if configs is None else configs):
        by_quant.setdefault(quant, {})[layout] = values
    return by_quant


def render(pending_note=True, configs=None, documents=None):
    configs = CONFIGS if configs is None else configs
    lines = [
        "Tokens/s. `Δ` compares the two RTX cards with one for the official and NVFP4 quants; "
        "EXL3 compares 2x RTX PRO 6000 with no Sparks against 1x RTX PRO 6000 capped at "
        "32 GiB plus 2x Spark, not isolated second-GPU scaling.",
        "",
        "Official columns are historical v6 measurements, not re-campaigned for v7. "
        "NVFP4 and EXL3 cells come from the selected v7 raw-result package, without historical fallback. "
        "Prefill headlines require a completed, passing campaign.",
        "",
        "| Measurement | Official 1x | Official 2x | Δ | NVFP4 1x | NVFP4 2x | Δ | EXL3 1x | EXL3 2x | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_quant = pairs(configs)
    for metric in METRICS:
        row = [metric]
        for quant in ("official", "nvfp4", "exl3"):
            one = by_quant[quant].get("1x", {}).get(metric)
            two = by_quant[quant].get("2x", {}).get(metric)
            row += [number(one), number(two), change(two, one)]
        lines.append("| " + " | ".join(row) + " |")
    if pending_note:
        missing = [f"{column} {metric.lower()}" for column, _q, _l, _p, values in configs
                   for metric in METRICS if values.get(metric) is None]
        if missing:
            lines += ["", "_No usable qualifying result: " + "; ".join(missing) +
                      ". Cells are marked — rather than estimated; records may be missing, unreadable, or incomplete._"]
    # Scientific/topology disclosure must not depend on showing the pending list.
    lines += ["", "_EXL3 1x uses one RTX PRO 6000 capped at 32 GiB including headroom plus two TP2 Sparks; "
                  "this simulates RTX 5090 memory capacity, not its performance. EXL3 2x uses no Sparks._"]
    for column, (decode, prefill) in (documents or {}).items():
        if prefill is not None and not prefill_complete(prefill):
            lines += ["", f"_{column} prefill has not established completed, passing status; "
                      "any available partial measurements are provisional and excluded from the headline._"]
        samples = (decode or {}).get("samples") or []
        if samples:
            passed = sum(s.get("passed") is True for s in samples)
            failed = sum(s.get("passed") is False for s in samples)
            unknown = len(samples) - passed - failed
            lines += ["", f"_{column}: {passed}/{len(samples)} decode sample checks explicitly passed; "
                      f"{failed} failed; {unknown} unknown/unreported. Throughput includes failed samples; "
                      "these checks do not establish a full benchmark pass. See the per-quant report for details._"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path,
                        default=Path.home() / ".cache/ds41rt-v7-package/performance")
    args = parser.parse_args()
    configs, documents = load_measurements(args.package)
    args.output.write_text(render(configs=configs, documents=documents))


if __name__ == "__main__":
    main()
