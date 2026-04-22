"""End-to-end tests: run textify() on asset images in every render mode."""

from pathlib import Path

import pytest
from PIL import Image

from proper_text_art.renderers import MODES
from proper_text_art.textify import textify


@pytest.fixture(name="sample_path")
def fixture_sample_path(assets: Path) -> Path:
    return assets / "blob" / "blob.png"


@pytest.mark.parametrize("mode", MODES)
def test_textify_ansi_runs_in_each_mode(sample_path: Path, mode: str) -> None:
    img = Image.open(sample_path)
    out = textify(img, mode=mode, output_format="ansi", num_colors=16)
    assert out, f"empty output for mode={mode}"
    assert out.count("\n") > 2, f"too few rows for mode={mode}"


def test_textify_html_emits_pre_block(sample_path: Path) -> None:
    img = Image.open(sample_path)
    out = textify(img, mode="blocks", output_format="html", num_colors=16)
    assert "<pre" in out
    assert "</pre>" in out


def test_textify_ascii_uses_edge_threshold(sample_path: Path) -> None:
    img = Image.open(sample_path)
    low = textify(
        img, mode="ascii", output_format="ansi", num_colors=16, edge_threshold=0.05
    )
    high = textify(
        img, mode="ascii", output_format="ansi", num_colors=16, edge_threshold=0.95
    )
    # Very low threshold → more directional chars; very high → pure ramp.
    directional = set("|/-\\")
    low_dir = sum(c in directional for c in low)
    high_dir = sum(c in directional for c in high)
    assert low_dir > high_dir
