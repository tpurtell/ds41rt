#!/usr/bin/env python3
"""Render the v7 headline table across all six configurations.

Three quants share two card topologies each, except EXL3 whose two
configurations are not a 1x/2x pair of one topology, so its change column is
computed the same way but read as "vs the single-card profile" rather than a
second-card gain.

Sources are named per cell so the table never implies a measurement that was
not taken: `official` cells come from the v6 release campaign (the official
image is not being re-campaigned), the NVFP4 and EXL3 cells come from the v7
package under ~/.cache/ds41rt-v7-package/performance.
"""
import argparse
import json
from pathlib import Path

# Each metric: label, and how to pull it from a config's records.
METRICS = [
    "Prefill",
    "Counting decode",
    "Weighted decode",
    "C1 code decode",
]

CONFIGS = [
    # (column, quant, layout, provenance, values)
    ("Official 1x", "official", "1x", "v6", {"Prefill": 7823.90, "Counting decode": 161.58,
                                             "Weighted decode": 92.00, "C1 code decode": 130.41}),
    ("Official 2x", "official", "2x", "v6", {"Prefill": 8355.22, "Counting decode": 221.64,
                                             "Weighted decode": 109.44, "C1 code decode": 155.70}),
    ("NVFP4 1x", "nvfp4", "1x", "v7", {"Prefill": None, "Counting decode": 113.3,
                                       "Weighted decode": 66.6, "C1 code decode": 88.8}),
    ("NVFP4 2x", "nvfp4", "2x", "v7", {"Prefill": 7432, "Counting decode": 152.25,
                                       "Weighted decode": 80.12, "C1 code decode": 109.2}),
    ("EXL3 1x", "exl3", "1x", "v7", {"Prefill": None, "Counting decode": None,
                                     "Weighted decode": None, "C1 code decode": None}),
    ("EXL3 2x", "exl3", "2x", "v7", {"Prefill": 5572, "Counting decode": 337.36,
                                     "Weighted decode": 150.64, "C1 code decode": 216.9}),
]


def number(value):
    return "—" if value is None else f"{value:,.0f}" if value >= 1000 else f"{value:,.2f}"


def change(second, first):
    if second is None or first is None:
        return "—"
    return f"{100 * (second / first - 1):+.1f}%"


def pairs():
    by_quant = {}
    for column, quant, layout, _prov, values in CONFIGS:
        by_quant.setdefault(quant, {})[layout] = values
    return by_quant


def render(pending_note=True):
    lines = [
        "Tokens/s. `Δ` compares the two RTX cards with one for the official and NVFP4 quants; "
        "for EXL3 it compares the two-card profile with the single-card one.",
        "",
        "| Measurement | Official 1x | Official 2x | Δ | NVFP4 1x | NVFP4 2x | Δ | EXL3 1x | EXL3 2x | Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        row = [metric]
        by_quant = pairs()
        for quant in ("official", "nvfp4", "exl3"):
            one = by_quant[quant].get("1x", {}).get(metric)
            two = by_quant[quant].get("2x", {}).get(metric)
            row += [number(one), number(two), change(two, one)]
        lines.append("| " + " | ".join(row) + " |")
    if pending_note:
        lines += ["", "_EXL3 single-card cells and the NVFP4 single-card prefill are still open; "
                      "they are marked — rather than estimated._"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path,
                        default=Path.home() / ".cache/ds41rt-v7-package/performance")
    args = parser.parse_args()
    # The measured v7 cells are read from the package when present so the table
    # and the raw results cannot drift apart.
    if args.package.is_dir():
        measured = {}
        for path in args.package.glob("*.json"):
            try:
                measured[path.stem] = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
        for column, quant, layout, prov, values in CONFIGS:
            if prov != "v7":
                continue
            decode = measured.get(f"{'single' if layout == '1x' else 'dual'}-{quant}-dspark")
            if decode and decode.get("samples"):
                import statistics
                per_case = {}
                for sample in decode["samples"]:
                    per_case.setdefault(sample["case"], []).append(
                        sample["observed_decode_tokens_per_second"])
                if "code" in per_case:
                    values["C1 code decode"] = round(statistics.median(per_case["code"]), 1)
                if "counting" in per_case:
                    values["Counting decode"] = round(statistics.median(per_case["counting"]), 1)
                weighted = decode.get("median_weighted_observed_decode_tokens_per_second")
                if weighted:
                    values["Weighted decode"] = round(weighted, 2)
            prefill = measured.get(f"{'single' if layout == '1x' else 'dual'}-{quant}-prefill")
            if prefill and prefill.get("cells"):
                best = max((c.get("median_effective_prefill_tokens_per_second") or 0)
                           for c in prefill["cells"])
                if best:
                    values["Prefill"] = round(best)
    args.output.write_text(render())


if __name__ == "__main__":
    main()
