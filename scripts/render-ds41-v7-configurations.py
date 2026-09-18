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
  * Spark per-layer share: measured, ~0.97 GB per layer per Spark.
  * NVFP4 and EXL3 expert bands, and every non-expert band: the runtime's own
    startup accounting (rank_peak_bytes, transport_bytes, reserved cache,
    headroom).
  * Speeds: the v6 release campaign for MXFP4, the v7 campaigns for NVFP4 and
    EXL3. A missing throughput renders as a dash, never as an estimate.
"""
import argparse
from pathlib import Path

BANDS = [
    ("Routed experts", "#4f8ef7"),
    ("KV + attention state", "#74e4c4"),
    ("Workspace + transport", "#b298ea"),
    ("Runtime + headroom", "#f2b45c"),
]
FREE = "#1b2b42"
CAPTION = [
    "Full height is device memory; colour-coded bands are measured use and the space above them is free.",
    "A dash is a configuration whose campaign has not run yet or whose profile is not implemented.",
]

# devices: (label, capacity_gb, {band: gb}); speeds: (label, code_tps, prefill_tps)
CONFIGS = [
    dict(
        title=["MXFP4 original", "1x RTX 6000 + 4x Spark"],
        subtitle="full-width layers on the card",
        devices=[
            ("RTX0", 96, {"Routed experts": 72.2, "KV + attention state": 16.0,
                          "Workspace + transport": 2.5, "Runtime + headroom": 2.4}),
            ("Spark x4", 128, {"Routed experts": 7.3, "KV + attention state": 2.0,
                               "Workspace + transport": 3.0, "Runtime + headroom": 1.0}),
        ],
        speeds=[("MXFP4", 130.4, 7824), ("NVFP4 W4A4", None, None)],
    ),
    dict(
        title=["MXFP4 original", "2x RTX 6000 + 4x Spark"],
        subtitle="20 TP2 layers on the cards",
        devices=[
            ("RTX0", 96, {"Routed experts": 72.2, "KV + attention state": 17.5,
                          "Workspace + transport": 2.0, "Runtime + headroom": 1.4}),
            ("RTX1", 96, {"Routed experts": 72.2, "KV + attention state": 17.5,
                          "Workspace + transport": 2.0, "Runtime + headroom": 1.4}),
            ("Spark x4", 128, {"Routed experts": 19.4, "KV + attention state": 2.0,
                               "Workspace + transport": 3.0, "Runtime + headroom": 1.0}),
        ],
        speeds=[("MXFP4", 155.7, 8355), ("NVFP4 W4A4", 109.2, 7432)],
    ),
    dict(
        title=["EXL3 2 bpw", "2x RTX 6000, no Spark"],
        subtitle="all 40 layers resident",
        devices=[
            ("RTX0", 96, {"Routed experts": 69.1, "KV + attention state": 17.0,
                          "Workspace + transport": 3.0, "Runtime + headroom": 1.5}),
            ("RTX1", 96, {"Routed experts": 69.1, "KV + attention state": 17.0,
                          "Workspace + transport": 3.0, "Runtime + headroom": 1.5}),
        ],
        speeds=[("EXL3 2 bpw", 216.9, 5572)],
    ),
    dict(
        title=["EXL3 2 bpw", "1x RTX 5090 + 2x Spark"],
        subtitle="32 GiB budget, degenerate",
        devices=[
            ("RTX0", 32, {"Routed experts": 12.0, "KV + attention state": 6.0,
                          "Workspace + transport": 4.0, "Runtime + headroom": 3.0}),
            ("Spark x2", 128, {"Routed experts": 52.0, "KV + attention state": 6.0,
                               "Workspace + transport": 8.0, "Runtime + headroom": 4.0}),
        ],
        speeds=[("EXL3 2 bpw", None, None)],
    ),
]

W, H = 1400, 900
TITLE_Y, CAPTION_Y, LEGEND_Y = 44, 70, 128
CARD_TOP, CARD_BOTTOM = 168, 872
BAR_TOP, BAR_AREA = 262, 452
BAR_W, BAR_GAP = 62, 24
MAX_DEVICE_GB = 128.0             # the tallest device fills BAR_AREA exactly
LEGEND_X, LEGEND_STEP = 60, 318   # one full-width row: four items, nothing clips


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
        for label, total, bands in config["devices"]:
            lines.append(bar(x, total, bands))
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{BAR_TOP + BAR_AREA + 24}" '
                         f'class="device" text-anchor="middle">{label}</text>')
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{BAR_TOP + BAR_AREA + 41}" '
                         f'class="sub" text-anchor="middle">{total} GB</text>')
            x += BAR_W + BAR_GAP
        head = BAR_TOP + BAR_AREA + 74
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
    stat_bottom = BAR_TOP + BAR_AREA + 74 + 24 + 23 * (rows - 1) + 8
    assert stat_bottom <= CARD_BOTTOM, f"stat block escapes its card ({stat_bottom} > {CARD_BOTTOM})"
    for config in CONFIGS:
        for label, total, bands in config["devices"]:
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
