#!/usr/bin/env python3
"""Render six serving profiles with explicit measured/estimated memory provenance."""
import argparse
from html import escape
import json
from pathlib import Path
import re
import runpy
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
W, H = 1440, 1760
BANDS = [("Experts / loading plan", "#4f8ef7"), ("KV", "#74e4c4"),
         ("Execution / transport", "#b298ea"), ("Other occupied", "#e7ad68")]
PUBLISHED = "docs/release-v7-published-memory-evidence.log"
COMPACT = "docs/release-v7-exl3-compact-evidence.json"
V6 = "docs/release-v6-performance.json"
PACK = "native/cuda/kernels/v41_expert_pack.cu:52-60"


def band(value, source, status="logged", basis=""):
    return dict(bytes=value, source=source, status=status, basis=basis)


def estimate(value, source, basis):
    return band(value, source, "estimate", basis)


def device(label, bands, occupied=None, occupancy_source=None, capacity=96, headroom=0):
    if occupied is not None:
        # A residual is not a measurement of attention or draft weights separately.
        bands.append(estimate(occupied - sum(b["bytes"] for b in bands), occupancy_source,
                              "occupied total minus the other displayed bands; includes backbone, draft, "
                              "snapshots, allocator/context and unclassified runtime; not component telemetry"))
    return dict(label=label, capacity=capacity, bands=bands, occupied=occupied,
                occupancy_source=occupancy_source, headroom=headroom)


def spark(weights, source, basis, execution=None):
    return device("Spark each", [estimate(weights, source, basis),
        band(0, "expert-only worker architecture", "structural", "KV lives on coordinator, not Spark"),
        execution or estimate(2 * GIB, "chart memory model", "heuristic 2 GiB execution/transport allowance; "
                             "not measured, excludes OS/shared RAM and is not a capacity guarantee")],
        capacity=128_000_000_000 / GIB)


def config(title, topology, key, label, devices, notes, sources):
    return dict(title=[title, topology], speed_columns=[(label, key)], devices=devices,
                notes=notes, sources=sources)


def configurations():
    result = []
    launches = json.loads((ROOT / V6).read_text())["launches"]
    for layout, count in [("single", 1), ("dual", 2)]:
        launch = launches[layout]
        dep = launch["deployment"]
        src = f"{V6}#/launches/{layout}"
        used = [int(v) * (1 << 20) for v in re.findall(r", (\d+) MiB,", launch["gpu_memory"])][:count]
        devices = []
        for rank in range(count):
            experts = 7219445760 * dep["rtx_expert_layers"] // count
            # KV rank split approximates asymmetric head placement; physical tails omitted.
            share = 1 if count == 1 else (0.6 if rank == 0 else 0.4)
            kv = int(dep["global_pool_bytes"] * share)
            devices.append(device(f"RTX{rank}", [
                estimate(experts, [src + "/deployment", PACK],
                         "384 experts * 5120 * ceil((2304/TP)/128)*128 * 51/32 bytes * local layers"),
                band(kv, src + "/deployment/global_pool_bytes", "logged" if count == 1 else "estimate",
                     "logical global pool only" if count == 1 else "global pool split 60:40; excludes private tails/padding"),
                band(0, src, "structural", "execution included in Other residual, not separately measured")],
                used[rank], src + "/gpu_memory", headroom=dep["runtime_headroom_bytes_per_gpu"]))
        devices.append(spark(2005401600 * dep["spark_layers"], [src + "/deployment/spark_layers", PACK],
                             "TP4 intermediate 576 padded to 640: 384*5120*640*51/32 bytes per loaded layer"))
        ram = launch["ram_capacity"]
        result.append(config("Official MXFP4 W4A8", f"{count}x RTX PRO 6000 + 4x Spark", f"Official {count}x",
            "MXFP4 (v6)", devices,
            ["Historical v6; est. experts use packed geometry.",
             f"Spark loaded/active layers: {dep['spark_layers']}/{dep['spark_active_layers']}; KV stays on RTX.",
             f"Host pinned: {ram['pinned_bytes']/GIB:.2f} GiB (not VRAM).",
             "Other est. = sampled used minus expert/KV bands."], [src, PACK]))
    one = device("RTX0", [band(30576500736, PUBLISHED + "#NVFP4-1x resident_bytes"),
        band(16745176576, PUBLISHED + "#historical-WIP cache_bytes (runs/v7q-a1/single-nvfp4-coordinator.log:4)", basis="WIP startup; same device token budget"),
        band(766661816, PUBLISHED + "#NVFP4-1x workspace_bytes")], 94073257984,
        PUBLISHED + "#NVFP4-1x ready_device_occupied_bytes", headroom=2 * GIB)
    result.append(config("NVFP4 W4A4", "1x RTX PRO 6000 + 4x Spark", "NVFP4 1x", "NVFP4",
        [one, spark(36 * 1911035904, PUBLISHED + "#NVFP4-1x Spark",
                    "36 loaded layers * logged 1911035904 bytes/layer")],
        ["Published startup; KV from matching WIP token budget.", "Other est. = ready occupancy minus logged bands.",
         "Spark exec est.: 2 GiB heuristic; excludes OS.", "Host pinned: 3.25 GiB (not VRAM)."], [PUBLISHED, "runs/v7q-a1/single-nvfp4-coordinator.log:4"]))
    dual = []
    for rank, (kv, transport, occupied) in enumerate(zip([2385404800, 1597086336],
            [1794720344, 2046378584], [98666020864, 101136465920])):
        dual.append(device(f"RTX{rank}", [band(76451266644, PUBLISHED + "#NVFP4-2x rank_peak_bytes", "plan",
            "expert loading peak, not persistent weights"), band(kv, PUBLISHED + "#NVFP4-2x cache_bytes"),
            band(transport, PUBLISHED + "#NVFP4-2x transport_bytes")], occupied,
            PUBLISHED + "#NVFP4-2x allocated_KV_occupied_bytes", headroom=838860800))
    dual.append(spark(20 * 1911035904, PUBLISHED + "#NVFP4-1x Spark; #NVFP4-2x expert_layers",
                      "40 minus 20 local layers, same TP4 bytes/layer as single; estimated placement"))
    result.append(config("NVFP4 W4A4", "2x RTX PRO 6000 + 4x Spark", "NVFP4 2x", "NVFP4", dual,
        ["Published startup; expert band is loading peak plan.", "Other est. subtracts plan, KV and transport from used.",
         "Spark exec est.: 2 GiB heuristic; excludes OS.", "Host pinned: 15.25 GiB (not VRAM)."], [PUBLISHED]))
    compact = device("RTX0 budget", [band(3433037824, PUBLISHED + "#EXL3 resident_bytes"),
        band(2191668736, PUBLISHED + "#EXL3 cache_bytes"), band(177724136, PUBLISHED + "#EXL3 workspace_bytes")],
        30428889088, PUBLISHED + "#EXL3 ready_device_occupied_bytes", capacity=32, headroom=2 * GIB)
    worker = spark(67394076672, COMPACT + "#/memory/spark_resident_bytes_each",
                   "historical same checkpoint, TP2 and 39 layers; published worker confirms topology",
                   estimate(799442308 + 2 * GIB, PUBLISHED + "#EXL3-historical-Spark (runs/v7q-a1/compact-ostrich.log:1,41)",
                            "capacity4096 historical workspace 799442308 + heuristic 2 GiB context/transport; excludes OS"))
    worker["bands"][0] = band(67394076672, COMPACT + "#/memory/spark_resident_bytes_each",
                               basis="historically logged TP2 weights; published topology matches, not a new measurement")
    result.append(config("EXL3 2.0 bpw", "1x RTX PRO 6000 + 2x TP2 Spark", "EXL3 5090+2-spark", "EXL3",
        [compact, worker], ["32 GiB includes headroom; NOT RTX 5090 hardware.",
         "RTX: published startup; other = residual est.", "Spark: historical weights; est. cap4096 workspace",
         "plus 2 GiB transport/context allowance, excluding OS."], [PUBLISHED, COMPACT, "runs/v7q-a1/compact-ostrich.log:1,41"]))
    log = json.loads((ROOT / COMPACT).read_text())["logs"]["dual-regression.log"]
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    occupied = json.loads(re.findall(r'allocated KV cache" occupied_bytes=(\[[^\]]+\])', log)[0])
    local = []
    for rank, (kv, transport) in enumerate(zip([7877077888, 5258201728], [2620852264, 2872510504])):
        src = COMPACT + "#/logs/dual-regression.log"
        local.append(device(f"RTX{rank}", [band(69122129920, src + " rank_peak_bytes", "plan", "expert loading peak"),
            band(kv, src + " cache_bytes"), band(transport, src + " transport_bytes")], occupied[rank],
            src + " occupied_bytes"))
    result.append(config("EXL3 2.0 bpw", "2x RTX PRO 6000, no Spark", "EXL3 2x6000 0-spark", "EXL3", local,
        ["All 40 layers local; historical rebuilt regression.", "Logged expert loading plan, KV and transport.",
         "Other est. = post-KV used minus those bands.", "Not published-image or campaign-peak telemetry."], [COMPACT + "#/logs/dual-regression.log"]))
    return result


CONFIGS = configurations()


def load_measurements(package):
    module = runpy.run_path(str(ROOT / "scripts/render-ds41-v7-headline.py"))
    rows, documents = module["load_measurements"](package)
    return {column: values for column, _q, _l, _p, values in rows}, documents


def render(package=None):
    package = Path.home() / ".cache/ds41rt-v7-published/performance" if package is None else Path(package)
    values, documents = load_measurements(package)
    metadata = dict(memory=CONFIGS, units="GiB = 2^30 bytes; Spark 128 GB nominal shared RAM; RTX 96 GiB nominal",
                    throughput=dict(package=str(package), v6=V6,
                        selector="scripts/render-ds41-v7-headline.py:load_measurements", values=values),
                    headroom="Configured reserve, not measured usage; included in compact 32 GiB ceiling")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
           '<title id="title">V7 serving configurations</title>',
           '<desc id="desc">Six profiles. Solid bands are logged allocations or explicit loading plans. Hatched bands and est. labels are estimates. Dark area is NOT measured free memory.</desc>',
           '<metadata>' + escape(json.dumps(metadata)) + '</metadata>',
           '<defs><pattern id="estimated" width="8" height="8" patternUnits="userSpaceOnUse"><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#091423" stroke-opacity=".55" stroke-width="2"/></pattern></defs>',
           '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#e4ecf8;font-size:13px}.title{font-size:26px;font-weight:bold}.panel{font-size:19px;font-weight:bold}.small{font-size:12px;fill:#b9c8dd}</style>',
           f'<rect width="{W}" height="{H}" fill="#091423"/>']

    def text(x, y, value, cls="small"):
        out.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(str(value))}</text>')

    text(24, 38, "V7 / six serving configurations", "title")
    text(24, 64, "Memory in GiB. Solid = logged allocation / plan; hatch + est. = estimate. Dark area is NOT measured free memory.")
    text(24, 84, "RTX outlines: 96 GiB nominal, except compact = 32 GiB including headroom. Spark: 128 GB shared RAM, one worker per bar.")
    text(24, 104, "Startup / historical memory and throughput are separate observations, not peak telemetry. Host pinned RAM is excluded from bars.")
    for i, (name, color) in enumerate(BANDS):
        x = 24 + i * 350
        out.append(f'<rect x="{x}" y="120" width="14" height="14" fill="{color}"/>')
        text(x + 22, 132, name)
    for index, cfg in enumerate(CONFIGS):
        left, top = 20 + (index % 2) * 710, 154 + (index // 2) * 514
        out.append(f'<rect class="panel-box" x="{left}" y="{top}" width="690" height="500" rx="10" fill="#0e1c2f" stroke="#35506c"/>')
        text(left + 16, top + 30, cfg["title"][0], "panel")
        text(left + 16, top + 54, cfg["title"][1], "panel")
        for j, dev in enumerate(cfg["devices"]):
            y = top + 78 + j * 83
            cap = dev["capacity"]
            text(left + 16, y, f'{dev["label"]} / {cap:.2f} GiB' +
                 (f' / occupied {dev["occupied"]/GIB:.2f} GiB' if dev["occupied"] is not None else ' / partial model'))
            x = left + 16
            width = cap / (128_000_000_000 / GIB) * 650
            out.append(f'<rect x="{x}" y="{y+8}" width="{width:.2f}" height="22" fill="#1b2b42" stroke="#536a83"/>')
            labels = []
            for b, (name, color) in zip(dev["bands"], BANDS):
                size = b["bytes"] / 128_000_000_000 * 650
                tip = escape(json.dumps(b))
                out.append(f'<rect x="{x:.2f}" y="{y+8}" width="{size:.2f}" height="22" fill="{color}"><title>{tip}</title></rect>')
                if b["status"] == "estimate":
                    out.append(f'<rect x="{x:.2f}" y="{y+8}" width="{size:.2f}" height="22" fill="url(#estimated)"/>')
                if b["bytes"] == 0 and name == "Execution / transport":
                    labels.append("Execution in Other")
                else:
                    labels.append(f'{name.split(" /")[0]} {b["bytes"]/GIB:.2f}' +
                                  (' est.' if b["status"] == 'estimate' else ' plan' if b["status"] == 'plan' else ''))
                x += size
            text(left + 16, y + 47, " | ".join(labels))
            if dev["headroom"]:
                text(left + 16, y + 63, f'Configured headroom: {dev["headroom"]/GIB:.2f} GiB (reserve, not occupied)')
        for row, note in enumerate(cfg["notes"]):
            text(left + 16, top + 338 + row * 18, note)
        label, column = cfg["speed_columns"][0]
        speed = values[column]
        fmt = lambda v: "—" if v is None else f"{v:,.0f}"
        text(left + 16, top + 430, f'{label} | code {fmt(speed.get("C1 code decode"))} | prefill {fmt(speed.get("Prefill"))} tok/s', "panel")
        samples = (documents.get(column, ({}, {}))[0] or {}).get("samples", [])
        if samples and any(s.get("passed") is not True for s in samples):
            text(left + 16, top + 454, f'{sum(s.get("passed") is True for s in samples)}/{len(samples)} decode checks; failed samples included')
        text(left + 16, top + 478, "Memory: " + ("historical v6" if index < 2 else "historical regression" if index == 5 else "published startup + disclosed estimates"))
    text(24, 1714, "— = no usable qualifying throughput result in the selected package. Prefill requires completed, passing status; no historical v7 fallback.")
    text(24, 1736, "Exact bytes, source fields and estimate formulas are embedded in metadata and band tooltips. Spark estimates exclude OS / unrelated shared RAM.")
    return "\n".join(out) + "\n</svg>\n"


def self_check(svg):
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    panels = root.findall('.//s:rect[@class="panel-box"]', ns)
    assert len(panels) == len(CONFIGS) == 6
    for index, panel in enumerate(panels):
        x, y = float(panel.get("x")), float(panel.get("y"))
        assert (x, y) == (20 + index % 2 * 710, 154 + index // 2 * 514)
        assert (float(panel.get("width")), float(panel.get("height"))) == (690, 500)
    for rect in root.findall(".//s:rect", ns):
        x, y, w, h = [float(rect.get(k, 0)) for k in ("x", "y", "width", "height")]
        assert min(x, y, w, h) >= 0 and x + w <= W + .01 and y + h <= H
    for text in root.findall(".//s:text", ns):
        assert 0 <= float(text.get("x")) <= W and 0 <= float(text.get("y")) <= H
    for cfg in CONFIGS:
        for dev in cfg["devices"]:
            assert sum(b["bytes"] for b in dev["bands"]) <= dev["capacity"] * GIB
            if dev["capacity"] == 32:
                assert sum(b["bytes"] for b in dev["bands"]) + dev["headroom"] <= 32 * GIB
            for b in dev["bands"]:
                assert b["bytes"] >= 0 and b["source"]
                assert b["status"] in ("logged", "plan", "structural", "estimate")
                assert b["status"] != "estimate" or b["basis"]
    headline = runpy.run_path(str(ROOT / "scripts/render-ds41-v7-headline.py"))
    assert [c["speed_columns"][0][1] for c in CONFIGS] == [c[0] for c in headline["CONFIGS"]]
    assert "NOT measured free" in svg and "NOT RTX 5090 hardware" in svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path.home() / ".cache/ds41rt-v7-published/performance")
    args = parser.parse_args()
    svg = render(args.package)
    self_check(svg)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
