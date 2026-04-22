"""Public API: turn a PIL image into text-mode art (ANSI or HTML)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from proper_text_art import colors, mesh, utils
from proper_text_art.pixelate import downsample
from proper_text_art.renderers import render_ansi, render_html
from proper_text_art.structure import compute_structure


@dataclass
class CellGrid:
    """Intermediate artifact shared by ANSI and HTML renderers for one image."""

    rgba: np.ndarray
    structure: np.ndarray | None


def compute_cell_grid(
    image: Image.Image,
    mode: str = "blocks",
    num_colors: int | None = None,
    initial_upscale_factor: int = 2,
    transparent_background: bool = False,
    pixel_width: int | None = None,
    intermediate_dir: Path | None = None,
) -> CellGrid:
    """Run the vision pipeline and return the per-cell data needed to render.

    Mirrors ``pixelate.pixelate()`` orchestration but stops before the final
    upscale (we want one char per cell, not many pixels per cell) and — for
    ``ascii`` mode — also returns the per-cell structure map from the scaled
    source image.
    """
    image_rgba = image.convert("RGBA")
    mesh_lines, upscale_factor = mesh.compute_mesh_with_scaling(
        image_rgba,
        initial_upscale_factor,
        output_dir=intermediate_dir,
        pixel_width=pixel_width,
    )

    skip_quantization = num_colors is None
    if skip_quantization:
        processed = image_rgba
    else:
        processed = colors.palette_img(
            image_rgba, num_colors=num_colors, output_dir=intermediate_dir
        )

    scaled_img = utils.scale_img(processed, upscale_factor)
    scaled_alpha = (
        None
        if skip_quantization
        else colors.extract_and_scale_alpha(image_rgba, upscale_factor)
    )

    grid = downsample(
        scaled_img,
        mesh_lines,
        skip_quantization=skip_quantization,
        original_alpha=scaled_alpha,
    )
    if transparent_background:
        grid = colors.make_background_transparent(grid)

    rgba = np.array(grid.convert("RGBA"))
    structure = compute_structure(scaled_img, mesh_lines) if mode == "ascii" else None
    return CellGrid(rgba=rgba, structure=structure)


def textify(
    image: Image.Image,
    mode: str = "blocks",
    output_format: str = "ansi",
    num_colors: int | None = None,
    initial_upscale_factor: int = 2,
    transparent_background: bool = False,
    pixel_width: int | None = None,
    ramp: str | None = None,
    edge_threshold: float = 0.15,
    intermediate_dir: Path | None = None,
) -> str:
    """Convert an image to text-mode art.

    Args:
        image: PIL image (any mode; it is converted to RGBA internally).
        mode: one of ``blocks``, ``half``, ``ascii``, ``shade``.
        output_format: ``ansi`` or ``html``.
        num_colors: palette quantization; ``None`` preserves all colors.
        initial_upscale_factor: forwarded to mesh detection.
        transparent_background: match the original ``pixelate`` flag.
        pixel_width: override automatic pixel-width detection.
        ramp: custom luminance-fallback ramp for ``ascii`` mode.
        edge_threshold: structural-char threshold for ``ascii`` mode.
        intermediate_dir: optional dir for pipeline debugging output.
    """
    grid = compute_cell_grid(
        image,
        mode=mode,
        num_colors=num_colors,
        initial_upscale_factor=initial_upscale_factor,
        transparent_background=transparent_background,
        pixel_width=pixel_width,
        intermediate_dir=intermediate_dir,
    )
    render = render_html if output_format == "html" else render_ansi
    return render(
        grid.rgba,
        mode=mode,
        structure=grid.structure,
        ramp=ramp,
        edge_threshold=edge_threshold,
    )
