#!/usr/bin/env python3
"""Render the two v7 per-quant performance reports from the measured package.

Both reports are generated from the bench JSONs under
~/.cache/ds41rt-v7-package/performance so a table can never disagree with the
raw results. A measurement that has not been taken renders as an em dash and is
listed at the end of its report; nothing is estimated.
"""
import argparse
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
    ("counting", "Counting 1-200"),
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
        layouts=[("2x", "dual", "2x RTX PRO 6000, no Spark",
                  "All 40 routed-expert layers resident as TP2 on the pair; no Spark worker.")],
        note="Uniform K=2 trellis with a two-tier kernel family; the paired 3.25 bpw "
             "publication uses the second tier.",
    ),
}

PENDING_NVFP4 = ["1x best prefill (campaign interrupted by a service restart)",
                 "1x and 2x tool-call evaluation runs"]
PENDING_EXL3 = ["1x compact profile (32 GiB card simulated, 2 Sparks) - not implemented; "
                "needs TP2 sharding on the expert side",
                "2x tool-call evaluation runs"]


def load(package: Path, stem: str):
    path = package / f"{stem}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def decode_cells(document):
    if not document or not document.get("samples"):
        return None
    per_case = {}
    for sample in document["samples"]:
        per_case.setdefault(sample["case"], []).append(sample["observed_decode_tokens_per_second"])
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


def best_prefill(document):
    values = [c.get("median_effective_prefill_tokens_per_second")
              for c in (document or {}).get("cells") or []]
    values = [value for value in values if value]
    return max(values) if values else None


def fmt(value, thousands=False):
    if value is None:
        return "—"
    return f"{value:,.0f}" if thousands or value >= 1000 else f"{value:,.2f}"


def change(second, first):
    if second is None or first is None:
        return "—"
    return f"{100 * (second / first - 1):+.1f}%"


def render(quant: str, package: Path) -> str:
    spec = QUANTS[quant]
    measured = {layout: (decode_cells(load(package, f"{stem}-{quant}-dspark")),
                         load(package, f"{stem}-{quant}-prefill"))
                for layout, stem, _, _ in spec["layouts"]}
    lines = [
        f"# DS41RT v7 performance - {spec['title']}",
        "",
        f"Checkpoint `{spec['checkpoint']}`. {spec['note']}",
        "",
        "Measured with the v7 release protocol: three samples per cell, 400 W per card, "
        "standard memory speed, reasoning code at high effort, all other throughput cases "
        "with thinking disabled, dSpark speculative decoding enabled for every decode cell.",
        "Each report table is generated from the raw bench output in the v7 package by "
        "`scripts/render-ds41-v7-quant-reports.py`; an em dash is a measurement not yet taken.",
        "",
        "## Configurations",
        "",
    ]
    for _layout, _stem, label, detail in spec["layouts"]:
        lines.append(f"- **{label}** — {detail}")
    lines += ["", "## Headline", "", "Tokens/s.", ""]
    header = ["Measurement"] + [label for _l, _s, label, _d in spec["layouts"]]
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
    lines += ["", "## Content-type decode", "", "Median tokens/s.", ""]
    header = ["Case"] + [label for _l, _s, label, _d in spec["layouts"] if measured[_l][0]]
    if len(header) > 1:
        lines += ["| " + " | ".join(header) + " |",
                  "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        for case, name in CASES:
            row = [name]
            for layout in [l for l, _s, _lb, _d in spec["layouts"] if measured[l][0]]:
                row.append(fmt(measured[layout][0][1].get(case)))
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Prefill", "",
              "Median effective tokens/s after shape warmup, by retained context and added tokens.", ""]
    for layout, _stem, label, _detail in spec["layouts"]:
        bases, suffixes, lookup = prefill_matrix(measured[layout][1])
        lines.append(f"**{label}**")
        lines.append("")
        if not bases:
            lines += ["_Not measured yet._", ""]
            continue
        lines += ["| Retained | " + " | ".join(f"+{s // 1024}K" for s in suffixes) + " |",
                  "|" + "|".join(["---"] + ["---:"] * len(suffixes)) + "|"]
        for base in bases:
            row = [f"{base // 1024}K"]
            row += [fmt(lookup.get((base, suffix)), thousands=True) for suffix in suffixes]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    pending = PENDING_NVFP4 if quant == "nvfp4" else PENDING_EXL3
    lines += ["## Not yet measured", ""]
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
