# Proper Text Art

## Summary

Converts pixel-art-style images (AI-generated or sourced from the web) into
**text-mode art**: ANSI-colored blocks, half-blocks, structural ASCII, or
Unicode shade blocks. It's a fork of
[proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art) — the
mesh-detection and per-cell color-selection pipeline is reused verbatim; the
final stage is swapped from "write a pixel" to "write a character".

Because the underlying algorithm already recovers the true pixel grid of a
noisy source image, every grid cell naturally becomes one character of output.

## Render modes

| Mode | Glyph(s) | What it does |
|---|---|---|
| `blocks` (default) | `█` | One full block per cell with a 24-bit foreground color. Highest fidelity to the source. |
| `half` | `▀` `▄` `█` | Packs two vertical cells into one character with fg = top, bg = bottom. Roughly 2× the vertical resolution of `blocks` at the same aspect ratio. |
| `ascii` | ` .:-=+*#%@` + `\| / - \\` | **Structural ASCII.** Sobel gradients pick a directional char for edge cells (`\|` vertical, `-` horizontal, `/` and `\\` diagonals); flat regions fall back to a luminance ramp. Still colored in truecolor. |
| `shade` | ` ░▒▓█` | Unicode shade blocks by luminance with a truecolor fg tint. |

Transparent cells (alpha < 128) are emitted as bare spaces.
Color output is always 24-bit truecolor (modern terminals support it — see
[ANSI color support](https://github.com/termstandard/colors#terminal-colors)).

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone and install:

```bash
git clone git@github.com:caioluders/proper-text-art.git
cd proper-text-art
uv sync
```

## Usage

### CLI (`pta`)

```bash
# Print colored block art to the terminal
uv run pta assets/blob/blob.png

# Save HTML you can open in a browser (extension infers the format)
uv run pta assets/blob/blob.png -m ascii -o art.html

# Half-block ANSI saved to a file, with quantization to 16 colors
uv run pta assets/ash/ash.png -m half -c 16 -o ash.txt

# Tune the ascii mode edge threshold (0..1): lower = more directional chars
uv run pta assets/demon/demon.png -m ascii --edge-threshold 0.08
```

#### Options

| Option | Description |
|---|---|
| INPUT (positional) | Source image file. |
| `-o`, `--output` `<path>` | Output file. If omitted, ANSI art is written to stdout. |
| `-m`, `--mode` `{blocks,half,ascii,shade}` | Render mode (default: `blocks`). |
| `-f`, `--format` `{ansi,html}` | Override output format; default is inferred from `-o` extension. |
| `--ramp` `<string>` | Custom luminance ramp for `ascii` mode (default: `" .:-=+*#%@"`). |
| `--edge-threshold` `<float>` | When `ascii` mode switches from ramp to directional chars (default: `0.15`). |
| `-c`, `--colors` `<int>` | Palette quantization (1–256). Omit to preserve all colors. |
| `-t`, `--transparent` | Make the dominant boundary color transparent. |
| `-w`, `--pixel-width` `<int>` | Override auto-detected pixel width. |
| `-u`, `--initial-upscale` `<int>` | Initial upscale factor for mesh detection (default: `2`). |

### Web UI (`pta-web`)

```bash
uv sync --extra web
uv run pta-web
# Opens http://127.0.0.1:7860
```

Upload an image, pick a mode, tweak the sliders. The preview pane renders the
HTML output; a code box shows the raw ANSI string for copy/paste.

### Python API

```python
from PIL import Image
from proper_text_art.textify import textify

img = Image.open("assets/blob/blob.png")

ansi = textify(img, mode="blocks", num_colors=16)
print(ansi)

html = textify(img, mode="ascii", output_format="html", edge_threshold=0.1)
open("art.html", "w").write(html)
```

## Structural ASCII — how it works

Naive ASCII art maps brightness to a ramp and calls it done, which smears
edges into dots. The `ascii` mode is edge-aware: for every mesh cell it
examines the *underlying pixels* (not just the collapsed color) and picks a
character that reflects the local gradient.

1. Run Sobel on the grayscale scaled source → `dx`, `dy` per pixel.
2. For each mesh cell, aggregate:
   - `edge_strength` = mean `hypot(dx, dy)` in the cell (normalized 0..1)
   - `edge_angle`    = `atan2(sum dy, sum dx)` — vector-summed to avoid the
     circular-mean artifact of averaging raw angles
   - `luminance`     = mean grayscale value (0..1)
3. Pick a char:
   - If `edge_strength > --edge-threshold`, bucket the angle into one of
     four directional chars (`|`, `/`, `-`, `\`). The gradient is perpendicular
     to the edge, so a roughly horizontal gradient (bright-to-dark left↔right)
     produces `|`, a vertical gradient produces `-`, and so on.
   - Otherwise fall back to `ramp[int(luminance * (len(ramp)-1))]`.

## Pipeline (unchanged from proper-pixel-art)

The mesh-detection and cell-color-selection algorithm is reused verbatim from
the upstream project. It takes a noisy, high-resolution pixel-art-style image
and recovers the true underlying grid:

1. Trim the border and zero out mostly-transparent pixels.
2. Upscale (default ×2) for stabler edge detection.
3. [Canny edge detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html).
4. [Morphological closing](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) to fill small gaps.
5. [Probabilistic Hough transform](https://docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html), keeping only near-vertical/horizontal lines.
6. Find the grid spacing (median of line differences, outlier-trimmed) and
   complete the mesh.
7. Quantize to N colors (skippable).
8. Pick one representative color per cell.

Once the cell grid is known, `proper-text-art` renders it as characters
instead of pixels. Credit for the vision pipeline goes to
[@KennethJAllen](https://github.com/KennethJAllen) — see
[proper-pixel-art](https://github.com/KennethJAllen/proper-pixel-art).

## License

MIT — same as upstream.
