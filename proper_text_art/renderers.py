r"""Render a cell-grid (RGBA + optional structure map) as ANSI or HTML text art.

Five modes:
- ``blocks`` — one ``█`` per cell, truecolor foreground.
- ``double`` — two ``██`` per cell, truecolor foreground. Doubles horizontal
  width so each pixel reads roughly square in a typical ~2:1 monospace cell.
- ``half``   — ``▀``/``▄``/``█`` with fg = top cell, bg = bottom cell; pairs rows
  to roughly double vertical resolution (character cells are ~2:1 tall).
- ``ascii``  — structural ASCII: ``| / - \`` for strong edges (oriented from the
  Sobel gradient), luminance ramp ``" .:-=+*#%@"`` for flat regions. Requires a
  ``structure`` map from :mod:`proper_text_art.structure`.
- ``shade``  — Unicode shade blocks ``" ░▒▓█"`` by luminance, truecolor fg tint.

Transparent cells (alpha < 128) emit a bare space with no escape codes.
Color output is 24-bit truecolor only.
"""

from __future__ import annotations

import math

import numpy as np

MODES = ("blocks", "double", "half", "ascii", "shade")

DEFAULT_RAMP = " .:-=+*#%@"
SHADE_RAMP = " ░▒▓█"  # " ░▒▓█"

FULL_BLOCK = "█"  # █
UPPER_HALF = "▀"  # ▀
LOWER_HALF = "▄"  # ▄

ALPHA_THRESHOLD = 128


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _ramp_char(lum: float, ramp: str) -> str:
    idx = int(round(lum * (len(ramp) - 1)))
    return ramp[max(0, min(len(ramp) - 1, idx))]


def _structural_char(
    edge_strength: float,
    edge_angle: float,
    luminance: float,
    ramp: str,
    edge_threshold: float,
) -> str:
    """Pick a char based on local gradient; fall back to luminance ramp."""
    if edge_strength <= edge_threshold:
        return _ramp_char(luminance, ramp)

    # Gradient is perpendicular to the edge direction.
    # Fold to [0, pi) since gradient(a) and gradient(a+pi) describe the same edge.
    angle = edge_angle % math.pi
    deg = math.degrees(angle)
    if deg < 22.5 or deg >= 157.5:
        return "|"  # horizontal gradient → vertical edge
    if deg < 67.5:
        return "/"  # anti-diagonal gradient → anti-diagonal edge
    if deg < 112.5:
        return "-"  # vertical gradient → horizontal edge
    return "\\"  # diagonal gradient → main-diagonal edge


def _is_transparent(cell: np.ndarray) -> bool:
    return int(cell[3]) < ALPHA_THRESHOLD


def _pick_char(
    cell: np.ndarray,
    mode: str,
    ramp: str,
    structure_cell: np.ndarray | None,
    edge_threshold: float,
) -> str:
    rgb = (int(cell[0]), int(cell[1]), int(cell[2]))
    if mode == "blocks":
        return FULL_BLOCK
    if mode == "double":
        return FULL_BLOCK * 2
    if mode == "shade":
        return _ramp_char(_luminance(rgb), SHADE_RAMP)
    if mode == "ascii":
        if structure_cell is None:
            raise ValueError("'ascii' mode requires a structure map")
        return _structural_char(
            float(structure_cell[0]),
            float(structure_cell[1]),
            float(structure_cell[2]),
            ramp,
            edge_threshold,
        )
    raise ValueError(f"unsupported mode: {mode!r}")


def _cell_pad(mode: str) -> str:
    """Padding emitted for transparent cells (matches the cell's char width)."""
    return "  " if mode == "double" else " "


# ---------------------------------------------------------------------------
# ANSI emitter
# ---------------------------------------------------------------------------

ANSI_RESET = "\x1b[0m"
ANSI_BG_DEFAULT = "\x1b[49m"


def _ansi_fg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _ansi_bg(rgb: tuple[int, int, int]) -> str:
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _render_ansi_half(grid: np.ndarray) -> str:
    h, w, _ = grid.shape
    lines: list[str] = []
    for j in range(0, h, 2):
        top = grid[j]
        bot = grid[j + 1] if j + 1 < h else np.zeros_like(top)
        bot_exists = j + 1 < h
        parts: list[str] = []
        # Tracks whether the previous cell left an ANSI bg color set; if so, a
        # following transparent (or top-transparent) cell must clear it,
        # otherwise the prior cell's bg leaks rightward into the empty area.
        bg_set = False
        for i in range(w):
            t_trans = _is_transparent(top[i])
            b_trans = not bot_exists or _is_transparent(bot[i])
            t_rgb = (int(top[i, 0]), int(top[i, 1]), int(top[i, 2]))
            b_rgb = (int(bot[i, 0]), int(bot[i, 1]), int(bot[i, 2]))
            if t_trans and b_trans:
                prefix = ANSI_BG_DEFAULT if bg_set else ""
                parts.append(f"{prefix} ")
                bg_set = False
            elif t_trans:
                prefix = ANSI_BG_DEFAULT if bg_set else ""
                parts.append(f"{prefix}{_ansi_fg(b_rgb)}{LOWER_HALF}")
                bg_set = False
            elif b_trans:
                prefix = ANSI_BG_DEFAULT if bg_set else ""
                parts.append(f"{prefix}{_ansi_fg(t_rgb)}{FULL_BLOCK}")
                bg_set = False
            else:
                parts.append(f"{_ansi_fg(t_rgb)}{_ansi_bg(b_rgb)}{UPPER_HALF}")
                bg_set = True
        lines.append("".join(parts) + ANSI_RESET)
    return "\n".join(lines)


def render_ansi(
    grid: np.ndarray,
    mode: str = "blocks",
    structure: np.ndarray | None = None,
    ramp: str | None = None,
    edge_threshold: float = 0.15,
) -> str:
    """Render the cell grid as a string of ANSI-colored characters."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "half":
        return _render_ansi_half(grid)

    ramp = ramp or DEFAULT_RAMP
    pad = _cell_pad(mode)
    h, w, _ = grid.shape
    lines: list[str] = []
    for j in range(h):
        parts: list[str] = []
        for i in range(w):
            cell = grid[j, i]
            if _is_transparent(cell):
                parts.append(pad)
                continue
            rgb = (int(cell[0]), int(cell[1]), int(cell[2]))
            sc = structure[j, i] if structure is not None else None
            ch = _pick_char(cell, mode, ramp, sc, edge_threshold)
            parts.append(f"{_ansi_fg(rgb)}{ch}")
        lines.append("".join(parts) + ANSI_RESET)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML emitter
# ---------------------------------------------------------------------------

_PRE_OPEN = (
    '<pre style="font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'
    '\'Liberation Mono\',monospace;line-height:1;background:#000;color:#fff;'
    'padding:1em;margin:0;white-space:pre;font-size:14px">'
)
_PRE_CLOSE = "</pre>"
_HTML_HEADER = (
    "<!doctype html>\n"
    '<meta charset="utf-8">\n'
    "<title>proper-text-art</title>\n"
    '<body style="background:#000;margin:0;padding:0">\n'
)
_HTML_FOOTER = "\n</body>\n"


def _html_escape(ch: str) -> str:
    if ch == "&":
        return "&amp;"
    if ch == "<":
        return "&lt;"
    if ch == ">":
        return "&gt;"
    return ch


def _html_span(ch: str, fg: tuple[int, int, int], bg: tuple[int, int, int] | None) -> str:
    style = f"color:rgb({fg[0]},{fg[1]},{fg[2]})"
    if bg is not None:
        style += f";background:rgb({bg[0]},{bg[1]},{bg[2]})"
    return f'<span style="{style}">{_html_escape(ch)}</span>'


def _render_html_half_body(grid: np.ndarray) -> str:
    h, w, _ = grid.shape
    lines: list[str] = []
    for j in range(0, h, 2):
        top = grid[j]
        bot = grid[j + 1] if j + 1 < h else np.zeros_like(top)
        bot_exists = j + 1 < h
        parts: list[str] = []
        for i in range(w):
            t_trans = _is_transparent(top[i])
            b_trans = not bot_exists or _is_transparent(bot[i])
            t_rgb = (int(top[i, 0]), int(top[i, 1]), int(top[i, 2]))
            b_rgb = (int(bot[i, 0]), int(bot[i, 1]), int(bot[i, 2]))
            if t_trans and b_trans:
                parts.append(" ")
            elif t_trans:
                parts.append(_html_span(LOWER_HALF, b_rgb, None))
            elif b_trans:
                parts.append(_html_span(FULL_BLOCK, t_rgb, None))
            else:
                parts.append(_html_span(UPPER_HALF, t_rgb, b_rgb))
        lines.append("".join(parts))
    return "\n".join(lines)


def render_html(
    grid: np.ndarray,
    mode: str = "blocks",
    structure: np.ndarray | None = None,
    ramp: str | None = None,
    edge_threshold: float = 0.15,
    full_document: bool = True,
) -> str:
    """Render the cell grid as HTML: a ``<pre>`` of colored ``<span>``s.

    If ``full_document`` is True (default) the output is a self-contained HTML
    page; if False, just the ``<pre>…</pre>`` fragment is returned (useful for
    embedding in Gradio's HTML component).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if mode == "half":
        body = _render_html_half_body(grid)
    else:
        ramp = ramp or DEFAULT_RAMP
        pad = _cell_pad(mode)
        h, w, _ = grid.shape
        lines: list[str] = []
        for j in range(h):
            parts: list[str] = []
            for i in range(w):
                cell = grid[j, i]
                if _is_transparent(cell):
                    parts.append(pad)
                    continue
                rgb = (int(cell[0]), int(cell[1]), int(cell[2]))
                sc = structure[j, i] if structure is not None else None
                ch = _pick_char(cell, mode, ramp, sc, edge_threshold)
                parts.append(_html_span(ch, rgb, None))
            lines.append("".join(parts))
        body = "\n".join(lines)

    pre = f"{_PRE_OPEN}{body}{_PRE_CLOSE}"
    if not full_document:
        return pre
    return f"{_HTML_HEADER}{pre}{_HTML_FOOTER}"
