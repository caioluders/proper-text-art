"""Unit tests for the text renderers."""

import numpy as np

from proper_text_art.renderers import (
    ANSI_BG_DEFAULT,
    ANSI_RESET,
    DEFAULT_RAMP,
    FULL_BLOCK,
    LOWER_HALF,
    SHADE_RAMP,
    UPPER_HALF,
    render_ansi,
    render_html,
)


def _synthetic_grid() -> np.ndarray:
    """3x3 RGBA grid: red / green / blue on the top row, a transparent cell in the middle."""
    grid = np.zeros((3, 3, 4), dtype=np.uint8)
    grid[0, 0] = (255, 0, 0, 255)
    grid[0, 1] = (0, 255, 0, 255)
    grid[0, 2] = (0, 0, 255, 255)
    grid[1, 0] = (255, 255, 255, 255)
    grid[1, 1] = (0, 0, 0, 0)  # transparent
    grid[1, 2] = (0, 0, 0, 255)
    grid[2, :] = (128, 128, 128, 255)
    return grid


def test_render_ansi_blocks_has_expected_fg_and_char():
    grid = _synthetic_grid()
    out = render_ansi(grid, mode="blocks")
    assert f"\x1b[38;2;255;0;0m{FULL_BLOCK}" in out
    assert f"\x1b[38;2;0;255;0m{FULL_BLOCK}" in out
    # 3 rows of output
    assert out.count("\n") == 2
    # Every row ends in reset
    for line in out.split("\n"):
        assert line.endswith(ANSI_RESET)


def test_render_ansi_transparent_cell_is_plain_space():
    grid = _synthetic_grid()
    out = render_ansi(grid, mode="blocks")
    row1 = out.split("\n")[1]
    # Middle cell is transparent → a literal space appears between white and black cells
    assert f"\x1b[38;2;255;255;255m{FULL_BLOCK} \x1b[38;2;0;0;0m{FULL_BLOCK}" in row1


def test_render_ansi_half_doubles_row_packing():
    grid = _synthetic_grid()
    out = render_ansi(grid, mode="half")
    # 3 rows → ceil(3/2) == 2 output lines
    assert out.count("\n") == 1
    # The pair (row 0 top, row 1 bottom) produces UPPER_HALF chars with fg+bg
    first = out.split("\n")[0]
    assert UPPER_HALF in first
    # Middle of pair row: top green / bottom transparent → full block with top fg
    assert f"\x1b[38;2;0;255;0m{FULL_BLOCK}" in first
    # Leftover row (row 2) pairs with an implicit transparent bottom → full block grey
    second = out.split("\n")[1]
    assert f"\x1b[38;2;128;128;128m{FULL_BLOCK}" in second
    # Top-transparent + bottom-opaque pairing does not appear in this grid,
    # but assert the LOWER_HALF codepoint resolves when we build one.
    grid2 = np.zeros((2, 1, 4), dtype=np.uint8)
    grid2[1, 0] = (10, 20, 30, 255)
    out2 = render_ansi(grid2, mode="half")
    assert f"\x1b[38;2;10;20;30m{LOWER_HALF}" in out2


def test_render_ansi_half_resets_bg_before_transparent_cell():
    """An opaque half-block followed by a transparent cell must clear the bg.

    Otherwise the prior cell's ANSI bg color persists across the trailing
    space and looks like the opaque pixel was duplicated rightward.
    """
    grid = np.zeros((2, 2, 4), dtype=np.uint8)
    grid[0, 0] = (10, 20, 30, 255)
    grid[1, 0] = (40, 50, 60, 255)
    # Column 1 is fully transparent.
    out = render_ansi(grid, mode="half")
    line = out.split("\n")[0]
    # Sanity: the opaque cell sets fg+bg+▀.
    assert f"\x1b[38;2;10;20;30m\x1b[48;2;40;50;60m{UPPER_HALF}" in line
    # The transparent cell that follows must explicitly reset the bg before
    # emitting its space — otherwise the (40,50,60) bg leaks rightward.
    assert f"{UPPER_HALF}{ANSI_BG_DEFAULT} " in line


def test_render_ansi_half_resets_bg_before_top_transparent_cell():
    """Top-transparent + bottom-opaque must also clear leaked bg."""
    grid = np.zeros((2, 2, 4), dtype=np.uint8)
    grid[0, 0] = (10, 20, 30, 255)
    grid[1, 0] = (40, 50, 60, 255)
    # Column 1: top transparent, bottom opaque blue.
    grid[1, 1] = (0, 0, 200, 255)
    out = render_ansi(grid, mode="half")
    line = out.split("\n")[0]
    # The lower-half block must be preceded by a bg reset.
    assert f"{ANSI_BG_DEFAULT}\x1b[38;2;0;0;200m{LOWER_HALF}" in line


def test_render_ansi_double_emits_two_full_blocks_per_cell():
    """``double`` mode renders ``██`` per cell so pixels look square in a terminal."""
    grid = np.zeros((1, 2, 4), dtype=np.uint8)
    grid[0, 0] = (255, 0, 0, 255)
    grid[0, 1] = (0, 0, 0, 0)  # transparent
    out = render_ansi(grid, mode="double")
    # Opaque red cell → fg sequence followed by two full blocks (no extra fg in between).
    assert f"\x1b[38;2;255;0;0m{FULL_BLOCK}{FULL_BLOCK}" in out
    # Transparent cell pads with two spaces (matches the cell's char width).
    assert f"{FULL_BLOCK}{FULL_BLOCK}  " in out
    assert out.endswith(ANSI_RESET)


def test_render_html_double_emits_two_full_blocks_per_cell():
    grid = np.zeros((1, 1, 4), dtype=np.uint8)
    grid[0, 0] = (10, 20, 30, 255)
    out = render_html(grid, mode="double", full_document=False)
    assert (
        f'<span style="color:rgb(10,20,30)">{FULL_BLOCK}{FULL_BLOCK}</span>' in out
    )


def test_render_ansi_shade_uses_ramp():
    grid = np.zeros((1, 2, 4), dtype=np.uint8)
    grid[0, 0] = (0, 0, 0, 255)  # black → first shade char (' ')
    grid[0, 1] = (255, 255, 255, 255)  # white → last shade char ('█')
    out = render_ansi(grid, mode="shade")
    assert SHADE_RAMP[0] in out
    assert SHADE_RAMP[-1] in out


def test_render_ansi_ascii_uses_structural_chars():
    """A cell with a strong horizontal gradient → '|' (vertical edge)."""
    grid = np.zeros((1, 1, 4), dtype=np.uint8)
    grid[0, 0] = (200, 200, 200, 255)
    structure = np.zeros((1, 1, 3), dtype=np.float32)
    structure[0, 0, 0] = 0.9  # strong edge
    structure[0, 0, 1] = 0.0  # gradient pointing horizontally → vertical edge
    structure[0, 0, 2] = 0.8
    out = render_ansi(grid, mode="ascii", structure=structure)
    assert "|" in out


def test_render_ansi_ascii_falls_back_to_ramp():
    grid = np.zeros((1, 1, 4), dtype=np.uint8)
    grid[0, 0] = (200, 200, 200, 255)
    structure = np.zeros((1, 1, 3), dtype=np.float32)
    structure[0, 0, 0] = 0.01  # below threshold
    structure[0, 0, 2] = 0.5
    out = render_ansi(grid, mode="ascii", structure=structure)
    # With luminance 0.5 the ramp index is round(0.5 * (len(ramp)-1))
    expected = DEFAULT_RAMP[round(0.5 * (len(DEFAULT_RAMP) - 1))]
    assert expected in out
    # None of the directional edge chars should appear in a flat cell
    for ch in "|/-\\":
        assert ch not in out


def test_render_ascii_requires_structure():
    grid = np.zeros((1, 1, 4), dtype=np.uint8)
    grid[0, 0] = (100, 100, 100, 255)
    try:
        render_ansi(grid, mode="ascii")
    except ValueError:
        return
    raise AssertionError("expected ValueError when 'ascii' mode is used without a structure map")


def test_render_html_contains_pre_and_span():
    grid = _synthetic_grid()
    out = render_html(grid, mode="blocks")
    assert out.startswith("<!doctype html>")
    assert '<span style="color:rgb(255,0,0)">' in out
    assert FULL_BLOCK in out


def test_render_html_fragment_mode():
    grid = _synthetic_grid()
    out = render_html(grid, mode="blocks", full_document=False)
    assert out.startswith("<pre")
    assert out.endswith("</pre>")
