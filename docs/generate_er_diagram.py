"""Generate docs/ifa_er_diagram.drawio.svg from the IFA data model.

The .drawio.svg format is:
- A valid SVG file (renders on GitHub without any plugin)
- The SVG root carries a ``content`` attribute containing HTML-escaped
  mxGraphModel XML, which the draw.io VS Code extension reads to restore
  the fully editable diagram.

Run with:
    python docs/generate_er_diagram.py
"""

from html import escape
from pathlib import Path

ROW_H = 26  # px height of each table row / header

# ---------------------------------------------------------------------------
# draw.io mxGraphModel helpers
# ---------------------------------------------------------------------------

def _entity_mx(eid: str, name: str, x: int, y: int, w: int,
               attrs: list[str], fill: str, stroke: str) -> str:
    """Return mxGraphModel XML for one swimlane entity and its attribute rows."""
    h = ROW_H + len(attrs) * ROW_H
    c_style = (
        f"swimlane;fontStyle=1;align=center;startSize={ROW_H};"
        f"fillColor={fill};strokeColor={stroke};"
    )
    a_style = (
        "text;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;spacingLeft=4;overflow=hidden;rotatable=0;"
    )
    xml = (
        f'<mxCell id="{eid}" value="{name}" style="{c_style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/>'
        f"</mxCell>"
    )
    for i, attr in enumerate(attrs):
        xml += (
            f'<mxCell id="{eid}_{i}" value="{attr}" style="{a_style}" '
            f'vertex="1" parent="{eid}">'
            f'<mxGeometry y="{ROW_H * (i + 1)}" width="{w}" height="{ROW_H}" as="geometry"/>'
            f"</mxCell>"
        )
    return xml


def _edge_mx(
    eid: str,
    src: str,
    tgt: str,
    start_arrow: str,
    end_arrow: str,
    label: str = "",
) -> str:
    """Return mxGraphModel XML for one ER edge with crow's-foot markers."""
    style = (
        "edgeStyle=entityRelationEdgeStyle;html=1;"
        f"startArrow={start_arrow};startFill=0;"
        f"endArrow={end_arrow};endFill=0;"
    )
    return (
        f'<mxCell id="{eid}" value="{label}" style="{style}" '
        f'edge="1" source="{src}" target="{tgt}" parent="1">'
        f'<mxGeometry relative="1" as="geometry"/>'
        f"</mxCell>"
    )


# ---------------------------------------------------------------------------
# SVG visual-rendering helpers
# ---------------------------------------------------------------------------

def _entity_svg(name: str, x: int, y: int, w: int,
                attrs: list[str], fill: str, stroke: str) -> str:
    """Return SVG markup for one entity table."""
    h = ROW_H + len(attrs) * ROW_H
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="white" stroke="{stroke}" stroke-width="1.5" rx="3"/>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{ROW_H}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" rx="3"/>',
        # Re-draw bottom edge of header as straight (rx only on top corners)
        f'<rect x="{x}" y="{y + ROW_H // 2}" width="{w}" height="{ROW_H // 2}" '
        f'fill="{fill}" stroke="none"/>',
        f'<line x1="{x}" y1="{y + ROW_H}" x2="{x + w}" y2="{y + ROW_H}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<text x="{x + w // 2}" y="{y + ROW_H // 2}" text-anchor="middle" '
        f'dominant-baseline="middle" font-weight="bold" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="12.5">{name}</text>',
    ]
    for i, attr in enumerate(attrs):
        ay = y + ROW_H * (i + 1)
        parts.append(
            f'<line x1="{x}" y1="{ay}" x2="{x + w}" y2="{ay}" '
            f'stroke="#ddd" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 6}" y="{ay + ROW_H // 2}" dominant-baseline="middle" '
            f'font-family="Segoe UI,Arial,sans-serif" font-size="11">{attr}</text>'
        )
    return "\n  ".join(parts)


def _edge_svg(
    d: str,
    start_label: str,
    sx: int,
    sy: int,
    end_label: str,
    ex: int,
    ey: int,
) -> str:
    """Return SVG markup for one edge with endpoint cardinality labels."""
    parts = [
        f'<path d="{d}" fill="none" stroke="#666" stroke-width="1.5" '
        f'marker-end="url(#arrowEnd)"/>'
    ]
    if start_label:
        safe_start = escape(start_label)
        parts.append(
            f'<text x="{sx}" y="{sy}" text-anchor="middle" '
            f'font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#555">'
            f"{safe_start}</text>"
        )
    if end_label:
        safe_end = escape(end_label)
        parts.append(
            f'<text x="{ex}" y="{ey}" text-anchor="middle" '
            f'font-family="Segoe UI,Arial,sans-serif" font-size="10" fill="#555">'
            f"{safe_end}</text>"
        )
    return "\n  ".join(parts)


# ---------------------------------------------------------------------------
# Entity and edge data
# ---------------------------------------------------------------------------

ENTITIES: list[tuple] = [
    # (id, name, x, y, w, attrs, fill, stroke)
    ("s0", "Scenario", 280, 80, 250, [
        "start_age: int",
        "end_age: int",
        "tax_free_pot: float",
        "main_dc_pot: float",
        "secondary_dc_pot: float",
        "secondary_dc_drawdown_age: int?",
        "baseline_spending: float?",
    ], "#dae8fc", "#6c8ebf"),
    ("dp0", "DbPension", 20, 20, 200, [
        "start_age: int",
        "annual_amount: float",
    ], "#d5e8d4", "#82b366"),
    ("dcp0", "DcPot", 20, 130, 200, [
        "drawdown_start_age: int",
        "initial_balance: float",
    ], "#d5e8d4", "#82b366"),
    ("ls0", "LumpSumEvent", 20, 240, 200, [
        "age: int",
        "amount: float",
    ], "#fff2cc", "#d6b656"),
    ("ss0", "SpendingStepEvent", 20, 350, 200, [
        "start_age: int",
        "extra_per_year: float",
        "end_age: int?",
    ], "#fff2cc", "#d6b656"),
    ("mc0", "MarketConfig", 620, 20, 220, [
        "mean_return: float",
        "std_return: float",
        "random_seed: int",
        "num_simulations: int",
    ], "#ffe6cc", "#d79b00"),
    ("r0", "Results", 620, 200, 220, [
        "ages: NDArray[int]",
        "total_balances: NDArray[float]",
        "dc_balances: NDArray[float]",
        "secondary_dc_balances: NDArray[float]",
        "tax_free_balances: NDArray[float]",
        "db_income: NDArray[float]",
        "total_withdrawals: NDArray[float]",
    ], "#f8cecc", "#b85450"),
    ("mcs0", "MonteCarloSummary", 620, 460, 220, [
        "ages: NDArray[int]",
        "paths: NDArray[float]",
    ], "#f8cecc", "#b85450"),
    ("tr0", "TaxRegime (enum)", 280, 390, 250, [
        "REST_OF_UK",
        "SCOTLAND",
    ], "#e1d5e7", "#9673a6"),
]

EDGES_MX: list[tuple[str, str, str, str]] = [
    # (id, source, target, source_marker, target_marker, label)
    ("e1", "dp0",  "s0",  "ERone", "ERmany", "1..*"),
    ("e2", "dcp0", "s0",  "ERone", "ERmany", "1..*"),
    ("e3", "ls0",  "s0",  "ERone", "ERmany", "1..*"),
    ("e4", "ss0",  "s0",  "ERone", "ERmany", "1..*"),
    ("e5", "s0",   "mc0", "ERone", "ERone",  "1 to 1"),
    ("e6", "s0",   "r0",  "ERone", "ERmany", "1..*"),
    ("e7", "r0",   "mcs0", "ERmany", "ERone", "*..1"),
    ("e8", "s0",   "tr0", "ERmany", "ERone", "*..1"),
]

# (SVG path d, start_label, sx, sy, end_label, ex, ey)
# Connection points calculated from entity geometry above.
EDGES_SVG: list[tuple[str, str, int, int, str, int, int]] = [
    # DbPension right(220,59)  → Scenario left(280,122)
    ("M220,59 L250,59 L250,122 L280,122",   "1", 228, 50, "*", 272, 116),
    # DcPot right(220,169) → Scenario left(280,163)
    ("M220,169 L250,169 L250,163 L280,163", "1", 228, 160, "*", 272, 158),
    # LumpSumEvent right(220,279) → Scenario left(280,205)
    ("M220,279 L250,279 L250,205 L280,205", "1", 228, 270, "*", 272, 210),
    # SpendingStepEvent right(220,402) → Scenario left(280,246)
    ("M220,402 L250,402 L250,246 L280,246", "1", 228, 393, "*", 272, 252),
    # Scenario right(530,110) → MarketConfig left(620,85)
    ("M530,110 L575,110 L575,85 L620,85",   "1", 542, 101, "1", 608, 80),
    # Scenario right(530,184) → Results left(620,304)
    ("M530,184 L575,184 L575,304 L620,304", "1", 542, 175, "*", 608, 298),
    # Results bottom(730,408) → MonteCarloSummary top(730,460)
    ("M730,408 L730,460",                   "*", 744, 416, "1", 744, 454),
    # Scenario bottom(405,288) → TaxRegime top(405,390)
    ("M405,288 L405,390",                   "*", 418, 296, "1", 418, 384),
]


# ---------------------------------------------------------------------------
# Build mxGraphModel XML and escape it
# ---------------------------------------------------------------------------

mx_parts = [
    '<mxGraphModel>',
    '<root>',
    '<mxCell id="0"/>',
    '<mxCell id="1" parent="0"/>',
]
for eid, name, x, y, w, attrs, fill, stroke in ENTITIES:
    mx_parts.append(_entity_mx(eid, name, x, y, w, attrs, fill, stroke))
for eid, src, tgt, start_arrow, end_arrow, label in EDGES_MX:
    mx_parts.append(_edge_mx(eid, src, tgt, start_arrow, end_arrow, label))
mx_parts += ["</root>", "</mxGraphModel>"]

content_attr = escape("".join(mx_parts), quote=True)

# ---------------------------------------------------------------------------
# Build SVG visual rendering
# ---------------------------------------------------------------------------

entity_blocks = "\n  ".join(
    _entity_svg(name, x, y, w, attrs, fill, stroke)
    for _, name, x, y, w, attrs, fill, stroke in ENTITIES
)
edge_blocks = "\n  ".join(
    _edge_svg(d, start_label, sx, sy, end_label, ex, ey)
    for d, start_label, sx, sy, end_label, ex, ey in EDGES_SVG
)

SVG = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="880" height="560"
     viewBox="0 0 880 560"
     content="{content_attr}">
  <defs>
    <marker id="arrowEnd" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#666"/>
    </marker>
  </defs>
  <!-- edges (drawn first so entities render on top) -->
  {edge_blocks}
  <!-- entities -->
  {entity_blocks}
</svg>
"""

out = Path(__file__).parent / "ifa_er_diagram.drawio.svg"
out.write_text(SVG, encoding="utf-8")
print(f"Written {out}")

comparison_out = Path(__file__).parent / "ifa_er_diagram_option2_comparison.drawio.svg"
comparison_out.write_text(SVG, encoding="utf-8")
print(f"Written {comparison_out}")
