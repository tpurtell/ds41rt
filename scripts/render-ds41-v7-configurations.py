#!/usr/bin/env python3
"""Render the v7 configuration graphic.

Four serving configurations in the order the release reads them: the MXFP4
original on one and two RTX cards with four Spark workers, then the EXL3
2 bpw compact pair (two cards, and the degenerate single simulated 5090 plus
two Sparks). Each configuration is a group of full-height device bars; the
coloured bands are that device's memory use and the space above them is free
capacity. One legend serves the chart.

Numbers live in CONFIGS. Sources:
  * Routed experts, MXFP4 at 2x: 20 layers x 3.61 GB (DUAL_ROUTED_LAYER_BYTES
    in scripts/select-release-gpus.py).
  * MXFP4 at 1x: full-width layers on one card, so a layer is twice the TP2
    layer; ~10 layers fill the same card budget.
  * Spark per-layer share: measured, ~0.97 GB per layer per Spark. Sparks hold
    routed-expert shards only - attention and the KV cache live on the RTX
    cards - so their bands carry no KV/attention segment.
  * NVFP4 and EXL3 expert bands, and every non-expert band: the runtime's own
    startup accounting (rank_peak_bytes, transport_bytes, reserved cache,
    headroom).
  * Speeds: the v6 release campaign for MXFP4, the v7 campaigns for NVFP4 and
    EXL3. A missing throughput renders as a dash, never as an estimate.
"""
import argparse
from pathlib import Path

# Only two bands, because only two numbers are actually measured: the routed
# expert residency the loader reports, and everything else the device occupies
# (attention, KV, workspace, runtime) as the remainder of logged occupancy.
# Earlier revisions split that remainder into guessed sub-bands; that produced
# a KV band on Sparks, which hold no attention and no KV cache.
BANDS = [
    ("Routed experts", "#4f8ef7"),
    ("KV cache", "#74e4c4"),
    ("dSpark draft", "#9fd85a"),
    ("Transport + workspace", "#b298ea"),
    ("Attention, shared, runtime", "#f2b45c"),
]
FREE = "#1b2b42"
CAPTION = [
    "Full height is device memory; bands come from the runtime's startup logs and the space above them is free.",
    "A dash is a measurement not yet taken; a blank bar is a device whose accounting is not logged yet.",
]

# devices: (label, capacity_gb, {band: gb}); speeds: (label, code_tps, prefill_tps)
CONFIGS = [
    dict(
        title=["Original / NVIDIA", "1x RTX 6000 + 4x Spark"],
        subtitle="NVFP4: 4 full-width layers",
        devices=[
            # Logged: 4 resident layers, resident_bytes 30.58 GB, workspace 0.34 GB.
            ("RTX0", 103, "96 GiB", {"Routed experts": 30.6, "Transport + workspace": 0.34}),
            ("Spark x4", 128, "128 GB", {}),
        ],
        speeds=[("MXFP4", 130.4, 7824), ("NVFP4 W4A4", 88.8, None)],
    ),
    dict(
        title=["Original / NVIDIA", "2x RTX 6000 + 4x Spark"],
        subtitle="NVFP4 startup logs",
        devices=[
            # Logged for the W4A4 run: rank_peak_bytes 76.45 GB per card plus
            # 19.6 GB of remaining occupancy; the Spark holds expert shards.
            # Logged per card and asymmetric: cache_bytes 2.70/1.81 GB,
            # transport 1.59/1.84 GB, dSpark lanes 19.2/418.3 MB per lane.
            ("RTX0", 103, "96 GiB", {"Routed experts": 76.5, "KV cache": 2.70,
                                     "dSpark draft": 0.04, "Transport + workspace": 1.59,
                                     "Attention, shared, runtime": 15.3}),
            ("RTX1", 103, "96 GiB", {"Routed experts": 76.5, "KV cache": 1.81,
                                     "dSpark draft": 0.84, "Transport + workspace": 1.84,
                                     "Attention, shared, runtime": 18.3}),
            ("Spark x4", 128, "128 GB", {}),
        ],
        speeds=[("MXFP4", 155.7, 8355), ("NVFP4 W4A4", 109.2, 7432)],
    ),
    dict(
        title=["EXL3 2 bpw", "2x RTX 6000, no Spark"],
        subtitle="all 40 layers resident",
        devices=[
            # Logged per card: rank peak 69.1 GB, cache 7.88/5.26 GB, transport
            # 2.62/2.87 GB, occupancy 89.75/93.02 GB.
            ("RTX0", 103, "96 GiB", {"Routed experts": 69.1, "KV cache": 7.88,
                                     "Transport + workspace": 2.62,
                                     "Attention, shared, runtime": 10.2}),
            ("RTX1", 103, "96 GiB", {"Routed experts": 69.1, "KV cache": 5.26,
                                     "Transport + workspace": 2.87,
                                     "Attention, shared, runtime": 15.8}),
        ],
        speeds=[("EXL3 2 bpw", 216.9, 5572)],
    ),
    dict(
        title=["EXL3 2 bpw", "1x RTX 5090 + 2x Spark"],
        subtitle="profile not implemented",
        devices=[("RTX0", 34, "32 GiB", {}), ("Spark x2", 128, "128 GB", {})],
        speeds=[("EXL3 2 bpw", None, None)],
    ),
]

W, H = 1400, 900
TITLE_Y, CAPTION_Y, LEGEND_Y = 44, 70, 128
CARD_TOP, CARD_BOTTOM = 168, 872
BAR_TOP, BAR_AREA = 258, 440
BAR_W, BAR_GAP = 62, 24
MAX_DEVICE_GB = 128.0             # the tallest device fills BAR_AREA exactly
LEGEND_X, LEGEND_STEP = 60, 256   # one full-width row: five items


def gb_to_px(gb):
    return gb * BAR_AREA / MAX_DEVICE_GB


def bar(x, total_gb, bands):
    """One device: capacity outline, bands stacked from the baseline up."""
    height = gb_to_px(total_gb)
    y0 = BAR_TOP + BAR_AREA - height
    out = [f'<rect x="{x:.1f}" y="{y0:.1f}" width="{BAR_W}" height="{height:.1f}" rx="5" '
           f'fill="{FREE}" stroke="#35506c" stroke-width="1.6"/>']
    cursor = BAR_TOP + BAR_AREA
    for name, colour in BANDS:
        value = bands.get(name, 0.0)
        if value <= 0:
            continue
        segment = gb_to_px(value)
        cursor -= segment
        out.append(f'<rect x="{x:.1f}" y="{cursor:.1f}" width="{BAR_W}" height="{segment:.1f}" '
                   f'fill="{colour}"/>')
        if segment >= 16:
            out.append(f'<text x="{x + BAR_W / 2:.0f}" y="{cursor + segment / 2 + 5:.0f}" '
                       f'class="band" text-anchor="middle">{value:.1f}</text>')
    return "\n".join(out)


def render():
    panel_w = (W - 80 - 3 * 22) / 4
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">DS41RT v7 serving configurations</title>',
        '<desc id="desc">Four serving configurations drawn as device memory bars: the MXFP4 '
        'original on one and two RTX cards with four Spark workers, and the EXL3 2 bpw quant on '
        'two RTX cards or one simulated 32 GiB RTX 5090 with two Spark workers. Colour-coded bands '
        'show memory use, with measured code-decode and best observed prefill throughput beneath '
        'each configuration.</desc>',
        '<defs><style>',
        'text{font-family:Arial,Helvetica,sans-serif;fill:#e4ecf8}',
        '.title{font-size:27px;font-weight:700}',
        '.panel{font-size:17px;font-weight:700}',
        '.sub{font-size:13px;fill:#8fa3bd}',
        '.device{font-size:14px;fill:#c7d6ea}',
        '.band{font-size:12px;fill:#08111f;font-weight:700}',
        '.legend{font-size:14px}',
        '.colhead{font-size:12px;fill:#8fa3bd}',
        '.speed{font-size:15px}',
        '.tps{font-size:16px;font-weight:700;fill:#74e4c4}',
        '.pending{font-size:15px;fill:#f2b45c}',
        '.detail{font-size:11px;fill:#9fd85a}',
        '</style></defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="#091423"/>',
        f'<text x="40" y="{TITLE_Y}" class="title">V7 · four configurations</text>',
    ]
    for row, text in enumerate(CAPTION):
        lines.append(f'<text x="40" y="{CAPTION_Y + row * 20}" class="sub">{text}</text>')

    legend_w = LEGEND_STEP * len(BANDS) + 40
    lines.append(f'<rect x="40" y="{LEGEND_Y - 24}" width="{legend_w}" height="38" rx="9" '
                 f'fill="#0e1c2f" stroke="#35506c" stroke-width="1.4"/>')
    for column, (name, colour) in enumerate(BANDS):
        lx = LEGEND_X + column * LEGEND_STEP
        lines.append(f'<rect x="{lx}" y="{LEGEND_Y - 11}" width="15" height="15" rx="3" fill="{colour}"/>')
        lines.append(f'<text x="{lx + 22}" y="{LEGEND_Y}" class="legend">{name}</text>')

    for index, config in enumerate(CONFIGS):
        left = 40 + index * (panel_w + 22)
        lines.append(f'<rect x="{left:.0f}" y="{CARD_TOP}" width="{panel_w:.0f}" '
                     f'height="{CARD_BOTTOM - CARD_TOP}" rx="11" fill="#0e1c2f" '
                     f'stroke="#35506c" stroke-width="1.4"/>')
        for row, text in enumerate(config["title"]):
            lines.append(f'<text x="{left + 16:.0f}" y="{CARD_TOP + 30 + row * 21}" class="panel">{text}</text>')
        lines.append(f'<text x="{left + 16:.0f}" y="{CARD_TOP + 74}" class="sub">{config["subtitle"]}</text>')
        count = len(config["devices"])
        group = count * BAR_W + (count - 1) * BAR_GAP
        x = left + (panel_w - group) / 2
        for label, total, capacity_text, bands in config["devices"]:
            lines.append(bar(x, total, bands))
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{BAR_TOP + BAR_AREA + 24}" '
                         f'class="device" text-anchor="middle">{label}</text>')
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{BAR_TOP + BAR_AREA + 41}" '
                         f'class="sub" text-anchor="middle">{capacity_text}</text>')
            # KV and the draft arena are small next to a card's full height, so
            # print their numbers rather than relying on a sliver of colour.
            for row, (name, short) in enumerate((("KV cache", "KV"), ("dSpark draft", "dS"))):
                if name in bands and bands[name] < 10:
                    lines.append(f'<text x="{x + BAR_W / 2:.0f}" '
                                 f'y="{BAR_TOP + BAR_AREA + 55 + row * 12}" '
                                 f'class="detail" text-anchor="middle">{short} {bands[name]:.2f}</text>')
            x += BAR_W + BAR_GAP
        head = BAR_TOP + BAR_AREA + 82
        lines.append(f'<text x="{left + 16:.0f}" y="{head}" class="colhead">quant</text>')
        lines.append(f'<text x="{left + panel_w - 116:.0f}" y="{head}" class="colhead" '
                     f'text-anchor="end">code</text>')
        lines.append(f'<text x="{left + panel_w - 20:.0f}" y="{head}" class="colhead" '
                     f'text-anchor="end">prefill</text>')
        for row, (label, code, prefill) in enumerate(config["speeds"]):
            y = head + 24 + row * 23
            lines.append(f'<text x="{left + 16:.0f}" y="{y}" class="speed">{label}</text>')
            for dx, value in ((panel_w - 116, code), (panel_w - 20, prefill)):
                if value is None:
                    lines.append(f'<text x="{left + dx:.0f}" y="{y}" class="pending" '
                                 f'text-anchor="end">—</text>')
                else:
                    lines.append(f'<text x="{left + dx:.0f}" y="{y}" class="tps" '
                                 f'text-anchor="end">{value:,.0f}</text>')
    return "\n".join(lines) + "\n</svg>\n"


def self_check(svg):
    """Fail loudly rather than shipping an overlapping or clipped chart."""
    import re
    rects = [(float(a), float(b), float(c), float(d)) for a, b, c, d in re.findall(
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"', svg)]
    texts = [(float(x), float(y)) for x, y in re.findall(
        r'<text x="([-\d.]+)" y="([-\d.]+)"', svg)]
    for x, y, w, h in rects:
        assert x >= 0 and y >= 0 and x + w <= W + 0.5 and y + h <= H + 0.5, f"rect off canvas {(x, y, w, h)}"
    for x, y in texts:
        assert 0 <= x <= W and 0 <= y <= H, f"text off canvas {(x, y)}"
    assert LEGEND_Y + 14 < CARD_TOP, "legend overlaps the card band"
    rows = max(len(c["speeds"]) for c in CONFIGS)
    stat_bottom = BAR_TOP + BAR_AREA + 82 + 24 + 23 * (rows - 1) + 8
    assert stat_bottom <= CARD_BOTTOM, f"stat block escapes its card ({stat_bottom} > {CARD_BOTTOM})"
    for config in CONFIGS:
        for label, total, _text, bands in config["devices"]:
            used = sum(bands.get(name, 0.0) for name, _ in BANDS)
            assert used <= total + 0.01, f"{config['title'][0]}/{label} uses {used} > {total} GB"
    assert abs(gb_to_px(MAX_DEVICE_GB) - BAR_AREA) < 0.001, "tallest device must fill the bar area"
    assert 40 + LEGEND_STEP * len(BANDS) + 40 <= W - 40, "legend row runs past the canvas"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    svg = render()
    self_check(svg)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
