#!/usr/bin/env python3
"""Render v7 configurations from raw throughput and explicitly sourced accounting.

Bands are nonoverlapping logged allocations/plans, NOT a complete VRAM census.
Never infer an attention/runtime residual or mark the unaccounted area as free.
The dual EXL3 log is a rebuilt regression startup, not campaign-peak telemetry.
"""
import argparse
from html import escape
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
BANDS = [("Expert weights / plan", "#4f8ef7"), ("KV allocation", "#74e4c4"),
         ("Execution / transport", "#b298ea")]
W, H = 1440, 970
BAR_TOP, BASELINE, BAR_AREA = 280, 680, 400
MAX_GIB = 128_000_000_000 / GIB
# Raw byte values; plans remain labelled plans rather than measured occupancy.
# Each source is a named captured log, not a guessed per-layer multiplication.
# distributed.rs:49-82 builds rank_peak_bytes from weight residency + loading
# transient, separately from transport_bytes at :302-331 and reserved KV.
# It is a loading plan, not a claim of simultaneous resident/peak device VRAM.
CONFIGS = [
    dict(title=["Official / NVIDIA", "1x RTX PRO 6000 + 4x Spark"],
         subtitle="Bands: NVFP4 startup only",
         devices=[("RTX0", 96, [30576500736, 16745176576, 766661816]),
                  ("Spark each (x4)", MAX_GIB, [None, None, None])],
         notes=["Full ready occupancy: not logged", "Spark accounting: not recorded here"],
         sources=[dict(path="runs/v7q-a1/single-nvfp4-coordinator.log", lines="4-5",
                       fields=["resident_bytes", "cache_bytes", "workspace_bytes"],
                       scope="NVFP4 startup; not historical MXFP4 memory")],
         speed_columns=[("MXFP4 (v6)", "Official 1x"), ("NVFP4", "NVFP4 1x")]),
    dict(title=["Official / NVIDIA", "2x RTX PRO 6000 + 4x Spark"],
         subtitle="VRAM accounting not established here",
         devices=[("RTX0", 96, [None, None, None]), ("RTX1", 96, [None, None, None]),
                  ("Spark each (x4)", MAX_GIB, [None, None, None])],
         notes=["No sourced complete startup log", "No inferred VRAM bands"],
         sources=[],
         speed_columns=[("MXFP4 (v6)", "Official 2x"), ("NVFP4", "NVFP4 2x")]),
    dict(title=["EXL3 2.0 bpw", "2x RTX PRO 6000, no Spark"],
         subtitle="All 40 layers local; regression startup",
         devices=[("RTX0", 96, [69122129920, 7877077888, 2620852264]),
                  ("RTX1", 96, [69122129920, 5258201728, 2872510504])],
         notes=["After KV: 90.94 / 91.53 GiB occupied", "Expert band is rank_peak_bytes plan"],
         sources=[dict(path="docs/release-v7-exl3-compact-evidence.json#/logs/dual-regression.log",
                       original_path="runs/v7q-a1/dual-regression.log", lines="23,26-27",
                       fields=["rank_peak_bytes", "cache_bytes", "transport_bytes", "occupied_bytes"],
                       scope="rebuilt regression startup, not campaign peak")],
         speed_columns=[("EXL3", "EXL3 2x6000 0-spark")]),
    dict(title=["EXL3 2.0 bpw compact", "1x RTX PRO 6000 + 2x Spark"],
         subtitle="32 GiB budget; NOT RTX 5090 hardware",
         devices=[("RTX0 budget", 32, [3433037824, 2191668736, 177724136]),
                  ("Spark each (x2)", MAX_GIB, [67394076672, None, 56007284])],
         notes=["RTX ready: 28.34 GiB; reserve: 2 GiB", "Sampled RTX peak: 28.88 GiB"],
         sources=[dict(path="docs/release-v7-exl3-compact-evidence.json#/logs/compact-final-residency.log",
                       original_path="runs/v7q-a1/compact-final-residency.log", lines="4-7",
                       fields=["resident_bytes", "cache_bytes", "workspace_bytes", "device_occupied_bytes"],
                       scope="startup; 32 GiB is a budget, not physical PRO 6000 capacity"),
                  dict(path="docs/release-v7-exl3-compact-evidence.json#/logs/compact-final-ostrich.log",
                       original_path="runs/v7q-a1/compact-final-ostrich.log", lines="1,41",
                       fields=["resident_bytes", "workspace_bytes"],
                       corroboration="runs/v7q-a1/compact-final-dodo.log:1,41",
                       scope="each Spark; excludes transport/context, no Spark KV")],
         speed_columns=[("EXL3", "EXL3 5090+2-spark")]),
]


def load_measurements(package):
    # Share the headline's raw-result selection and completed-prefill gate.
    module = runpy.run_path(str(ROOT / "scripts/render-ds41-v7-headline.py"))
    rows, documents = module["load_measurements"](package)
    return {column: values for column, _q, _l, _p, values in rows}, documents


def render(package=None):
    package = Path.home() / ".cache/ds41rt-v7-package/performance" if package is None else Path(package)
    values, documents = load_measurements(package)
    provenance = dict(memory=[c["sources"] for c in CONFIGS],
                      throughput=dict(v7_package=str(package),
                                      v6="docs/release-v6-performance.json",
                                      selector="scripts/render-ds41-v7-headline.py:load_measurements",
                                      metric="median code decode; best completed passing cell median prefill"))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           '<title id="title">DS41RT v7 serving configurations</title>',
           '<desc id="desc">Logged partial allocation bands in GiB, not total VRAM or free space. '
           'Official throughput is historical v6; NVFP4 and EXL3 throughput is v7 WIP. '
           'Compact measurements use an RTX PRO 6000 with a 32 GiB ceiling, not a physical RTX 5090. '
           'Release Docker image publication remains pending.</desc>',
           '<metadata>' + escape(json.dumps(provenance)) + '</metadata>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#e4ecf8} '
           '.title{font-size:26px;font-weight:bold}.panel{font-size:16px;font-weight:bold} '
           '.small{font-size:12px;fill:#aebdd3}.note{font-size:13px;fill:#aebdd3} '
           '.band{font-size:12px;fill:#08111f;font-weight:bold} '
           '.value{font-size:15px;font-weight:bold;fill:#74e4c4}</style>',
           f'<rect width="{W}" height="{H}" rx="18" fill="#091423"/>']

    def text(x, y, value, cls="note", anchor="start"):
        out.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(str(value))}</text>')

    text(30, 40, "V7 · serving configurations", "title")
    text(30, 66, "Bands: directly logged allocations/plans (GiB). Dark area is unaccounted, NOT measured free memory.")
    text(30, 87, "Outlines: nominal capacity, except compact RTX = 32 GiB budget. Spark bars represent one worker each (128 GB nominal).")
    text(30, 108, "Memory logs and throughput campaigns are separate observations; startup allocation is not peak VRAM.")
    for i, (name, color) in enumerate(BANDS):
        x = 30 + 350 * i
        out.append(f'<rect x="{x}" y="126" width="16" height="16" fill="{color}"/>')
        text(x + 24, 140, name)
    text(30, 164, "WIP builds only: v7 release Docker images not built or published. RTX 5090 not hardware-qualified.")
    for index, config in enumerate(CONFIGS):
        left = 20 + 355 * index
        out.append(f'<rect x="{left}" y="188" width="335" height="687" rx="10" fill="#0e1c2f" stroke="#35506c"/>')
        for row, title in enumerate(config["title"]):
            text(left + 12, 216 + row * 22, title, "panel")
        text(left + 12, 262, config["subtitle"], "small")
        count = len(config["devices"])
        step = 100 if count == 3 else 128
        start = left + (335 - (count - 1) * step) / 2
        for j, (label, capacity, bands) in enumerate(config["devices"]):
            x = start + j * step
            height = capacity / MAX_GIB * BAR_AREA
            out.append(f'<rect x="{x-26}" y="{BASELINE-height:.2f}" width="52" height="{height:.2f}" fill="#1b2b42" stroke="#536a83"/>')
            cursor = BASELINE
            for raw, (name, color) in zip(bands, BANDS):
                if raw is None:
                    continue
                size = raw / GIB / MAX_GIB * BAR_AREA
                cursor -= size
                out.append(f'<rect x="{x-26}" y="{cursor:.2f}" width="52" height="{size:.2f}" fill="{color}"><title>{escape(name)}: {raw:,} bytes</title></rect>')
                if size > 20:
                    text(x, cursor + size / 2 + 4, f"{raw/GIB:.2f}", "band", "middle")
            text(x, 701, label, "small", "middle")
            text(x, 718, f"{capacity:.2f} GiB", "small", "middle")
        for row, note in enumerate(config["notes"]):
            text(left + 12, 742 + row * 17, note, "small")
        text(left + 12, 784, "tok/s", "small")
        text(left + 228, 784, "code", "small", "end")
        text(left + 322, 784, "prefill", "small", "end")
        for row, (label, column) in enumerate(config["speed_columns"]):
            y = 808 + row * 24
            text(left + 12, y, label)
            for x, metric in [(left + 228, "C1 code decode"), (left + 322, "Prefill")]:
                value = values[column].get(metric)
                text(x, y, "—" if value is None else f"{value:,.0f}", "value", "end")
        failures = []
        for _label, column in config["speed_columns"]:
            samples = (documents.get(column, ({}, {}))[0] or {}).get("samples", [])
            if samples and any(s.get("passed") is not True for s in samples):
                failures.append(f"{sum(s.get('passed') is True for s in samples)}/{len(samples)} decode checks; failed samples included")
        if failures:
            text(left + 12, 857, "; ".join(failures), "small")
    text(30, 904, "— = no usable qualifying result, not proof a measurement was never attempted. NVFP4 1x prefill was interrupted.")
    text(30, 926, "EXL3 reasoning-limit failures mean decode campaigns are not fully passing. See the per-quant reports and compact qualification evidence.")
    text(30, 948, "Exact source paths/fields are embedded in SVG metadata; no attention, draft-weight or runtime residuals are inferred.")
    return "\n".join(out) + "\n</svg>\n"


def self_check(svg):
    import re
    import xml.etree.ElementTree as ET
    ET.fromstring(svg)
    for x, y, width, height in re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg):
        x, y, width, height = map(float, (x, y, width, height))
        assert x >= 0 and y >= 0 and x + width <= W and y + height <= H
    for config in CONFIGS:
        for _label, capacity, bands in config["devices"]:
            assert sum(v or 0 for v in bands) / GIB <= capacity
    assert "NOT measured free" in svg and "NOT RTX 5090 hardware" in svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path.home() / ".cache/ds41rt-v7-package/performance")
    args = parser.parse_args()
    svg = render(args.package)
    self_check(svg)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
