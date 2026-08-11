"""Shared figure helpers for analysis plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    MODEL_DISPLAY_COLORS,
    MODEL_LABEL_MAP,
    MODEL_NAMES,
)

_MODEL_INDEX = {name: i for i, name in enumerate(MODEL_NAMES)}

# Setup markers / sizes for linked FP–FN (and similar) scatters.
SETUP_MARKER: dict[str, str] = {
    "Baseline": "o",
    "H": "s",
    "HL": "X",
    "HC": "^",
}
SETUP_SCATTER_S: dict[str, int] = {
    "Baseline": 68,
    "H": 68,
    "HL": 72,
    "HC": 68,
}

# Grouped bar styling (accuracy-by-k).
BAR_HATCHES = ["", ".", "/", "x"]
BAR_FILL_ALPHAS = [0.85, 0.35, 0.35, 0.35]


def model_label(model: str) -> str:
    """Display name for a model folder, falling back to its last path segment."""
    return MODEL_LABEL_MAP.get(str(model), str(model).split("/")[-1])


def model_color(model: str) -> str:
    idx = _MODEL_INDEX.get(str(model))
    if idx is None:
        return plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    return MODEL_DISPLAY_COLORS[idx % len(MODEL_DISPLAY_COLORS)]


def sort_models(models) -> list[str]:
    """Order by ``MODEL_ORDER``; unknown names sort last, alphabetically."""
    return sorted(
        {str(m) for m in models},
        key=lambda m: (_MODEL_INDEX.get(m, len(MODEL_NAMES)), m.lower()),
    )


def setup_label(setup: str, *, k: int | None = None) -> str:
    """Human-readable setup name; optionally annotate with ``k``."""
    if k is None:
        return str(setup)
    return f"{setup} (k={k})"


def setup_slug(setup: str) -> str:
    """Filesystem-safe setup token for figure / CSV stems."""
    return str(setup).lstrip("_") or "baseline"


def despine(ax) -> None:
    """Hide the top and right spines."""
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 240,
    also_pdf: bool = False,
    dpi_pdf: int | None = None,
    **savefig_kw,
) -> Path:
    """Save a figure, echo the path, optionally write a sibling PDF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", **savefig_kw}
    fig.savefig(path, dpi=dpi, **kw)
    print(f"Wrote {path}")
    if also_pdf:
        pdf = path.with_suffix(".pdf")
        pdf_kw = {k: v for k, v in kw.items() if k != "dpi"}
        if dpi_pdf is not None:
            pdf_kw["dpi"] = dpi_pdf
        fig.savefig(pdf, **pdf_kw)
        print(f"Wrote {pdf}")
    return path


def draw_baselines(
    ax,
    baselines: dict[str, float] | None,
    *,
    scale: float = 100.0,
    fontsize: float = 19.0,
    label_min_gap: float | None = None,
    color: str = "#aaa",
    label_color: str = "#555555",
    linestyle: str = ":",
    linewidth: float = 0.8,
    label_side: str = "right",
) -> None:
    """Dotted reference lines with dark-gray labels outside the right spine.

    When baselines sit within a point of each other (typical ~50), label
    y-positions are nudged apart so each remains readable. Gap scales with
    *fontsize* (about one line-height in data units on a 40-point y-range).
    """
    ordered = sorted(
        ((str(label), float(acc) * scale) for label, acc in (baselines or {}).items()),
        key=lambda item: item[1],
    )
    if not ordered:
        return

    for _, y in ordered:
        ax.axhline(
            y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=1
        )

    raw_ys = [y for _, y in ordered]
    y0, y1 = ax.get_ylim()
    ylim_ready = y0 <= min(raw_ys) and max(raw_ys) <= y1
    y_span = max(float(y1 - y0), 1.0) if ylim_ready else 40.0
    gap = (
        float(label_min_gap)
        if label_min_gap is not None
        else max(1.35, fontsize * y_span / 420.0)
    )

    label_ys = list(raw_ys)
    for i in range(1, len(label_ys)):
        label_ys[i] = max(label_ys[i], label_ys[i - 1] + gap)
    # Re-center the stack on the original cluster so lines stay nearby.
    mid_raw = 0.5 * (raw_ys[0] + raw_ys[-1])
    mid_new = 0.5 * (label_ys[0] + label_ys[-1])
    label_ys = [y - (mid_new - mid_raw) for y in label_ys]
    # Keep labels inside the axes once ylim is set (with a small pad).
    if ylim_ready:
        pad = 0.5 * gap
        if label_ys[0] < y0 + pad:
            delta = (y0 + pad) - label_ys[0]
            label_ys = [y + delta for y in label_ys]
        if label_ys[-1] > y1 - pad:
            delta = label_ys[-1] - (y1 - pad)
            label_ys = [y - delta for y in label_ys]

    if label_side == "left":
        label_x, ha = -0.01, "right"
    else:
        label_x, ha = 1.01, "left"

    for (label, y), y_lab in zip(ordered, label_ys):
        ax.text(
            label_x,
            y_lab,
            f"{label}: {y:.1f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha=ha,
            fontsize=fontsize,
            color=label_color,
            clip_on=False,
            zorder=6,
        )


def padded_limits(
    values: list[float],
    *,
    floor: float,
    ceil: float,
    pad: float = 1.0,
) -> tuple[float, float]:
    """``(floor, ceil)`` widened so nothing in *values* is clipped out of view."""
    if not values:
        return floor, ceil
    return min(floor, min(values) - pad), max(ceil, max(values) + pad)


def rate_axis_limits(
    xs: list[float],
    ys: list[float],
    *,
    lo: float = 0.0,
    hi: float = 100.0,
    pad_frac: float = 0.06,
    pad_min: float = 0.85,
    pad_max: float = 12.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Padded ``(xlim, ylim)`` for rate percentages, clamped to ``[lo, hi]``."""
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    px = float(np.clip((hi_x - lo_x) * pad_frac, pad_min, pad_max))
    py = float(np.clip((hi_y - lo_y) * pad_frac, pad_min, pad_max))
    return (
        (max(lo, lo_x - px), min(hi, hi_x + px)),
        (max(lo, lo_y - py), min(hi, hi_y + py)),
    )


def bar_face_edge(
    color: str,
    series_index: int,
    *,
    alphas: list[float] | None = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Face / edge RGBA for a grouped bar at *series_index*."""
    fills = alphas if alphas is not None else BAR_FILL_ALPHAS
    r, g, b, _ = mcolors.to_rgba(color)
    face = (r, g, b, fills[series_index % len(fills)])
    edge = (r * 0.75, g * 0.75, b * 0.75, 0.9)
    return face, edge
