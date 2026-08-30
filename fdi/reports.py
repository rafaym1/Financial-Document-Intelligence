import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patheffects import withSimplePatchShadow

from fdi.numbers import parse_scaled_number

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def parse_fiscal_year(fiscal_period: str) -> int | None:
    """Extract the fiscal year a fact's period string refers to, e.g. 'Year Ended December 31, 2025' -> 2025."""
    years = [int(match.group()) for match in YEAR_PATTERN.finditer(fiscal_period)]
    return max(years) if years else None


def build_metric_series(facts: list[dict], metric_name: str) -> dict[int, float]:
    """Collect (year -> value) for every fact whose metric name exactly matches (case-insensitive)."""
    series: dict[int, float] = {}
    for fact in facts:
        if fact["metric"].strip().lower() != metric_name.strip().lower():
            continue
        year = parse_fiscal_year(fact["fiscal_period"])
        value = parse_scaled_number(fact["value"])
        if year is not None and value is not None:
            series[year] = value
    return series


SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_COLOR = "#2a78d6"


def _format_compact_dollars(value: float) -> str:
    """Format a raw dollar value as a compact, human-readable figure, e.g. 1324144000.0 -> '$1.32B'."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.0f}"


def _shade(hex_color: str, amount: float) -> str:
    """Lighten (amount > 0) or darken (amount < 0) a hex color by blending toward white/black."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    if amount >= 0:
        r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + amount)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_metric_trend(metric_name: str, series: dict[int, float], output_path: Path) -> None:
    """Render a gradient-filled, drop-shadowed column chart of a metric's year-over-year values and save it as a PNG."""
    years = sorted(series)
    values = [series[year] for year in years]
    y_max = max(values) * 1.18

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bar_width = 0.45
    gradient_cmap = LinearSegmentedColormap.from_list(
        "bar_gradient", [_shade(SERIES_COLOR, -0.35), _shade(SERIES_COLOR, 0.35)]
    )
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)

    for x, value in enumerate(values):
        left, right = x - bar_width / 2, x + bar_width / 2

        # An invisible rectangle purely to cast a shadow -- the shadow offset is in screen
        # points, not data units, so it stays a consistent size regardless of the huge
        # x (few bars) vs. y (hundreds of millions) scale mismatch.
        shadow_rect = Rectangle((left, 0), bar_width, value, facecolor="none", edgecolor="none", zorder=2)
        shadow_rect.set_path_effects([withSimplePatchShadow(offset=(2, -3), shadow_rgbFace="#000000", alpha=0.18)])
        ax.add_patch(shadow_rect)

        # The visible fill: a vertical gradient image clipped to the bar's rectangle.
        clip_rect = Rectangle((left, 0), bar_width, value, transform=ax.transData)
        image = ax.imshow(
            gradient, extent=[left, right, 0, value], origin="lower", aspect="auto", cmap=gradient_cmap, zorder=3
        )
        image.set_clip_path(clip_rect)

        ax.text(
            x, value + y_max * 0.02, _format_compact_dollars(value),
            ha="center", va="bottom", fontsize=10, color=PRIMARY_INK, zorder=4,
        )

    ax.set_xlim(-0.6, len(years) - 0.4)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(year) for year in years], color=MUTED_INK, fontsize=10)

    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=2)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    ax.set_title(metric_name, color=PRIMARY_INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
