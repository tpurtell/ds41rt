#!/usr/bin/env python3
"""Render the v7 configuration graphic.

Four serving configurations, each as vertical device bars whose colour-coded
bands are that device's memory use, plus the measured code-decode and best
observed prefill throughput. One legend serves the whole chart.

Numbers live in CONFIGS below so a campaign result drops straight in.
Values are GB (10^9 bytes) as reported by the runtime's own accounting, or
MiB/1024 from nvidia-smi where noted; `None` renders as a pending cell and is
called out in the caption, never silently invented.
"""
import argparse
from pathlib import Path

# Band colours, shared by every bar; the legend is drawn once.
BANDS = [
    ("Routed experts", "#4f8ef7"),
    ("KV + attention state", "#74e4c4"),
    ("Workspace + transport", "#b298ea"),
    ("Runtime + headroom", "#f2b45c"),
]
CAPTION = "Bands are measured memory in use; the space above them is free. Dashed bars await a campaign."
FREE = "#1b2b42"

# Each config: title, subtitle, devices [(name, total_gb, bands)], code, prefill,
# note, measured. Bands come from the runtime's own startup accounting:
#   routed experts      rank_peak_bytes (weights + load staging)
#   KV + attention      reserved cache plus the attention/shared/runtime residual
#   workspace+transport transport_bytes
#   runtime + headroom  setup headroom
# Unmeasured configs render as dashed outlines so the topology is visible while
# the throughput cells stay honestly blank.
CONFIGS = [
    dict(
        title=["NVFP4 W4A4", "1× RTX + 4× Spark"],
        subtitle="same quant, one card",
        devices=[("RTX0", 96, {}), ("Spark", 128, {})],
        code=None, prefill=None, note="campaign pending", measured=False,
    ),
    dict(
        title=["NVFP4 W4A4", "2× RTX + 4× Spark"],
        subtitle="20 + 20 layers",
        devices=[
            ("RTX0", 96, {"Routed experts": 76.5, "KV + attention state": 16.9,
                          "Workspace + transport": 1.6, "Runtime + headroom": 1.0}),
            ("RTX1", 96, {"Routed experts": 76.5, "KV + attention state": 16.9,
                          "Workspace + transport": 1.8, "Runtime + headroom": 0.1}),
            ("Spark", 128, {"Routed experts": 19.4, "KV + attention state": 2.0,
                            "Workspace + transport": 3.0, "Runtime + headroom": 1.0}),
        ],
        code=109.2, prefill=7432, note="full campaign", measured=True,
    ),
    dict(
        title=["EXL3 2 bpw", "2× RTX"],
        subtitle="all 40 layers, no Spark",
        devices=[
            ("RTX0", 96, {"Routed experts": 69.1, "KV + attention state": 17.0,
                          "Workspace + transport": 3.0, "Runtime + headroom": 1.5}),
            ("RTX1", 96, {"Routed experts": 69.1, "KV + attention state": 17.0,
                          "Workspace + transport": 3.0, "Runtime + headroom": 1.5}),
        ],
        code=216.9, prefill=5572, note="quick campaign", measured=True,
    ),
    dict(
        title=["EXL3 2 bpw", "1× RTX 5090 + 2× Spark"],
        subtitle="32 GiB budget, degenerate",
        devices=[("RTX0", 32, {}), ("Spark", 128, {})],
        code=None, prefill=None, note="profile pending", measured=False,
    ),
]

W, H = 1400, 900
GB_PER_PX = 0.68          # bar scale: 96 GB card ~ 141 px
BAR_W, BAR_GAP = 58, 26
CARD_TOP_OFFSET, CARD_BOTTOM = 30, 712
PANEL_TOP, PANEL_H = 182, 500


def bar(x, top, total_gb, bands, measured=True):
    """One device: outline, stacked bands from the baseline up, GB labels."""
    height = total_gb / GB_PER_PX
    y0 = top + PANEL_H - height
    stroke = '#35506c' if measured else '#5a6b83'
    dash = '' if measured else ' stroke-dasharray="6 5"'
    out = [f'<rect x="{x}" y="{y0:.1f}" width="{BAR_W}" height="{height:.1f}" rx="6" '
           f'fill="{FREE}" stroke="{stroke}" stroke-width="2"{dash}/>']
    cursor = top + PANEL_H
    for name, colour in BANDS:
        value = bands.get(name, 0)
        if value <= 0:
            continue
        segment = value / GB_PER_PX
        cursor -= segment
        out.append(f'<rect x="{x}" y="{cursor:.1f}" width="{BAR_W}" height="{segment:.1f}" '
                   f'fill="{colour}"/>')
        if segment >= 15:
            out.append(f'<text x="{x + BAR_W / 2:.0f}" y="{cursor + segment / 2 + 5:.0f}" '
                       f'class="band" text-anchor="middle">{value:.1f}</text>')
    return "\n".join(out)


def render():
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">DS41RT v7 serving configurations</title>',
        '<desc id="desc">Four v7 serving configurations drawn as device memory bars: the NVFP4 '
        'W4A4 quant on one or two RTX cards with four Spark workers, and the EXL3 2 bpw quant on '
        'two RTX cards or one simulated 32 GiB RTX 5090 with two Spark workers. Coloured bands '
        'show each device\'s memory use, with measured code-decode and best observed prefill '
        'throughput beneath.</desc>',
        '<defs><style>',
        'text{font-family:Arial,Helvetica,sans-serif;fill:#e4ecf8}',
        '.title{font-size:28px;font-weight:700}',
        '.panel{font-size:19px;font-weight:700}',
        '.sub{font-size:14px;fill:#8fa3bd}',
        '.device{font-size:15px;fill:#c7d6ea}',
        '.band{font-size:13px;fill:#08111f;font-weight:700}',
        '.speed{font-size:16px}',
        '.speedbig{font-size:21px;font-weight:700;fill:#74e4c4}',
        '.legend{font-size:15px}',
        '.pending{font-size:15px;fill:#f2b45c}',
        '</style></defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="#091423"/>',
        '<text x="40" y="50" class="title">V7 · four configurations</text>',
        '<text x="40" y="74" class="sub">Bar height is device memory; bands are measured use, the remainder is free.</text>',
        f'<text x="40" y="96" class="sub">{CAPTION}</text>',
    ]
    panel_w = (W - 80 - 3 * 24) / 4
    legend_x = 700
    for index, config in enumerate(CONFIGS):
        left = 40 + index * (panel_w + 24)
        lines.append(f'<rect x="{left:.0f}" y="{PANEL_TOP - CARD_TOP_OFFSET}" width="{panel_w:.0f}" '
                     f'height="{CARD_BOTTOM}" rx="12" fill="#0e1c2f" stroke="#35506c" stroke-width="1.5"/>')
        for row, text in enumerate(config["title"]):
            lines.append(f'<text x="{left + 18:.0f}" y="{PANEL_TOP - 14 + row * 20}" class="panel">{text}</text>')
        lines.append(f'<text x="{left + 18:.0f}" y="{PANEL_TOP + 32}" class="sub">{config["subtitle"]}</text>')
        count = len(config["devices"])
        group = count * BAR_W + (count - 1) * BAR_GAP
        x = left + (panel_w - group) / 2
        for device, total, bands in config["devices"]:
            lines.append(bar(x, PANEL_TOP + 48, total, bands, config["measured"]))
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{PANEL_TOP + 48 + PANEL_H + 26:.0f}" '
                         f'class="device" text-anchor="middle">{device}</text>')
            lines.append(f'<text x="{x + BAR_W / 2:.0f}" y="{PANEL_TOP + 48 + PANEL_H + 44:.0f}" '
                         f'class="sub" text-anchor="middle">{total} GB</text>')
            x += BAR_W + BAR_GAP
        base = PANEL_TOP + 48 + PANEL_H + 80
        code = "—" if config["code"] is None else f'{config["code"]:.0f}'
        prefill = "—" if config["prefill"] is None else f'{config["prefill"]:,}'
        lines.append(f'<text x="{left + 18:.0f}" y="{base}" class="speed">code decode '
                     f'<tspan class="speedbig">{code}</tspan> tok/s</text>')
        lines.append(f'<text x="{left + 18:.0f}" y="{base + 22}" class="speed">best prefill '
                     f'<tspan class="speedbig">{prefill}</tspan> tok/s</text>')
        # An unmeasured half still reads as pending, never as a finished cell.
        note_class = "sub" if (config["measured"] and config["code"] is not None
                               and config["prefill"] is not None) else "pending"
        lines.append(f'<text x="{left + 18:.0f}" y="{base + 44}" class="{note_class}">{config["note"]}</text>')
    # One legend, top-right of the chart area.
    ly = 128
    step = 168
    lines.append(f'<rect x="{legend_x - 14}" y="{ly - 24}" width="{step * len(BANDS) + 6}" '
                 f'height="40" rx="10" fill="#0e1c2f" stroke="#35506c" stroke-width="1.5"/>')
    for column, (name, colour) in enumerate(BANDS):
        lx = legend_x + column * step
        lines.append(f'<rect x="{lx}" y="{ly - 12}" width="16" height="16" rx="3" fill="{colour}"/>')
        lines.append(f'<text x="{lx + 24}" y="{ly}" class="legend">{name}</text>')
    return "\n".join(lines) + "\n</svg>\n"


def self_check(svg):
    """Assert the layout invariants the reviewers flagged: nothing leaves the
    canvas, the stat block stays inside its card, and the legend never overlaps
    a card or the title block."""
    import re
    rects = [(float(a), float(b), float(c), float(d)) for a, b, c, d in re.findall(
        r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"', svg)]
    texts = [(float(x), float(y)) for x, y in re.findall(
        r'<text x="([-\d.]+)" y="([-\d.]+)"', svg)]
    for x, y, w, h in rects:
        assert x >= 0 and y >= 0 and x + w <= W + 0.5 and y + h <= H + 0.5, f"rect off canvas {(x, y, w, h)}"
    for x, y in texts:
        assert 0 <= x <= W and 0 <= y <= H, f"text off canvas {(x, y)}"
    cards = [(40 + i * (panel_w() + 24), PANEL_TOP - CARD_TOP_OFFSET,
              panel_w(), CARD_BOTTOM) for i in range(len(CONFIGS))]
    stat_note_y = PANEL_TOP + 48 + PANEL_H + 80 + 44
    assert stat_note_y + 8 <= cards[0][1] + CARD_BOTTOM, "stat block escapes its card"
    legend = (700 - 14, 128 - 24, 168 * len(BANDS) + 6, 40)
    for cx, cy, cw, chh in cards:
        overlap = not (legend[0] + legend[2] <= cx or cx + cw <= legend[0]
                       or legend[1] + legend[3] <= cy or cy + chh <= legend[1])
        assert not overlap, "legend overlaps a card"
    caption_ink_end = 40 + 96 * 6.6
    assert caption_ink_end < legend[0], "caption runs under the legend"


def panel_w():
    return (W - 80 - 3 * 24) / 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    svg = render()
    self_check(svg)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
