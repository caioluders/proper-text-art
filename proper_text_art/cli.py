"""Command-line interface for the ``pta`` text-mode art renderer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from proper_text_art.renderers import MODES, render_ansi, render_html
from proper_text_art.textify import compute_cell_grid


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pta",
        description=(
            "Render an image as text-mode art (ANSI-colored blocks, half-blocks, "
            "structural ASCII, or Unicode shade blocks)."
        ),
    )
    parser.add_argument(
        "input_path", type=Path, nargs="?", help="Path to the source image."
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path_flag",
        type=Path,
        help="Path to the source image (alias for the positional argument).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_path",
        type=Path,
        default=None,
        help=(
            "Output file. If omitted, ANSI art is written to stdout. "
            "If the extension is .html the format defaults to HTML, "
            "otherwise ANSI."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=MODES,
        default="blocks",
        help="Text render mode (default: blocks).",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=("ansi", "html"),
        default=None,
        help="Override output format; default is inferred from -o extension.",
    )
    parser.add_argument(
        "--ramp",
        dest="ramp",
        type=str,
        default=None,
        help=(
            "Custom luminance ramp for the 'ascii' mode fallback "
            "(default: ' .:-=+*#%%@')."
        ),
    )
    parser.add_argument(
        "--edge-threshold",
        dest="edge_threshold",
        type=float,
        default=0.15,
        help=(
            "Edge-strength threshold (0..1) at which 'ascii' mode switches from "
            "the luminance ramp to directional chars (default: 0.15)."
        ),
    )

    pipeline = parser.add_argument_group("Pipeline options")
    pipeline.add_argument(
        "-c",
        "--colors",
        dest="num_colors",
        type=int,
        default=None,
        help=(
            "Number of colors to quantize the image to (1-256). "
            "Omit to skip quantization and preserve all colors."
        ),
    )
    pipeline.add_argument(
        "-t",
        "--transparent",
        dest="transparent",
        action="store_true",
        default=False,
        help="Make the dominant boundary color transparent in the output.",
    )
    pipeline.add_argument(
        "-w",
        "--pixel-width",
        dest="pixel_width",
        type=int,
        default=None,
        help="Width of the 'pixels' in the input; auto-detected if omitted.",
    )
    pipeline.add_argument(
        "-u",
        "--initial-upscale",
        dest="initial_upscale",
        type=int,
        default=2,
        help="Initial upscale factor for mesh detection (default: 2).",
    )
    return parser


def _infer_format(out_path: Path | None, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if out_path is not None and out_path.suffix.lower() in (".html", ".htm"):
        return "html"
    return "ansi"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.input_path is None and args.input_path_flag is None:
        parser.error("You must provide an input path (positional or with -i).")
    args.input_path = args.input_path or args.input_path_flag
    args.output_format = _infer_format(args.out_path, args.output_format)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    img = Image.open(Path(args.input_path).expanduser())
    grid = compute_cell_grid(
        img,
        mode=args.mode,
        num_colors=args.num_colors,
        initial_upscale_factor=args.initial_upscale,
        transparent_background=args.transparent,
        pixel_width=args.pixel_width,
    )

    render = render_html if args.output_format == "html" else render_ansi
    rendered = render(
        grid.rgba,
        mode=args.mode,
        structure=grid.structure,
        ramp=args.ramp,
        edge_threshold=args.edge_threshold,
    )

    if args.out_path is None:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
        return

    out = Path(args.out_path).expanduser()
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
