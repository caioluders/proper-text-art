"""Per-cell structure features (edge strength, edge angle, luminance).

Used by the ``ascii`` render mode to pick a character that reflects the local
gradient of the underlying pixels — not just the collapsed per-cell color.
"""

import cv2
import numpy as np
from PIL import Image

from proper_text_art.utils import Mesh


def compute_structure(scaled_img: Image.Image, mesh_lines: Mesh) -> np.ndarray:
    """Compute per-cell structure features from the already-scaled source image.

    The returned array has shape ``(H, W, 3)`` where ``(H, W)`` matches the cell
    grid dimensions (``len(lines_y) - 1`` by ``len(lines_x) - 1``):

    - ``[..., 0]`` edge_strength — mean gradient magnitude in the cell, 0..1
    - ``[..., 1]`` edge_angle    — gradient direction in radians, -pi..pi
    - ``[..., 2]`` luminance     — mean grayscale value in the cell, 0..1

    Angles are aggregated as vectors (``atan2(sum dy, sum dx)``) to avoid the
    circular-mean artifact of averaging raw angles.

    Args:
        scaled_img: The image already scaled to match ``mesh_lines`` coordinates
            — i.e. the same ``scaled_img`` fed into ``pixelate.downsample``.
        mesh_lines: ``(x_lines, y_lines)`` from the mesh detector.
    """
    gray = np.asarray(scaled_img.convert("L"), dtype=np.float32)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(dx, dy)

    lines_x, lines_y = mesh_lines
    height, width = len(lines_y) - 1, len(lines_x) - 1
    out = np.zeros((height, width, 3), dtype=np.float32)

    # Sobel magnitude for 8-bit input tops out at ~4*255 in practice.
    # Divide by that to normalize into ~[0, 1].
    mag_norm = 4.0 * 255.0

    for j in range(height):
        y0, y1 = lines_y[j], lines_y[j + 1]
        if y1 <= y0:
            continue
        for i in range(width):
            x0, x1 = lines_x[i], lines_x[i + 1]
            if x1 <= x0:
                continue
            cell_mag = mag[y0:y1, x0:x1]
            cell_dx = dx[y0:y1, x0:x1]
            cell_dy = dy[y0:y1, x0:x1]
            cell_gray = gray[y0:y1, x0:x1]

            out[j, i, 0] = min(float(cell_mag.mean()) / mag_norm, 1.0)
            sx = float(cell_dx.sum())
            sy = float(cell_dy.sum())
            out[j, i, 1] = float(np.arctan2(sy, sx)) if (sx or sy) else 0.0
            out[j, i, 2] = float(cell_gray.mean()) / 255.0

    return out
