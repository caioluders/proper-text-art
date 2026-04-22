"""Gradio web interface for Proper Text Art."""

from __future__ import annotations

from PIL import Image

from proper_text_art.renderers import MODES, render_ansi, render_html
from proper_text_art.textify import compute_cell_grid

IMG_HEIGHT = 420


def process(
    image: Image.Image | None,
    mode: str,
    num_colors: int,
    transparent: bool,
    initial_upscale: int,
    pixel_width: int,
    edge_threshold: float,
    ramp: str,
) -> tuple[str, str]:
    """Run the pipeline once, render both HTML and ANSI from the same grid."""
    if image is None:
        return "", ""
    grid = compute_cell_grid(
        image,
        mode=mode,
        num_colors=num_colors if num_colors > 0 else None,
        initial_upscale_factor=initial_upscale,
        transparent_background=transparent,
        pixel_width=pixel_width if pixel_width > 0 else None,
    )
    ramp_arg = ramp.strip() if ramp and ramp.strip() else None
    html = render_html(
        grid.rgba,
        mode=mode,
        structure=grid.structure,
        ramp=ramp_arg,
        edge_threshold=edge_threshold,
        full_document=False,
    )
    ansi = render_ansi(
        grid.rgba,
        mode=mode,
        structure=grid.structure,
        ramp=ramp_arg,
        edge_threshold=edge_threshold,
    )
    return html, ansi


def create_demo():
    """Create Gradio demo interface."""
    import gradio as gr

    with gr.Blocks(title="Proper Text Art") as demo:
        gr.Markdown(
            "# Proper Text Art\n"
            "Convert pixel-art-style images to text-mode art "
            "(ANSI-colored Unicode or structural ASCII)."
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(
                    type="pil",
                    label="Input",
                    format="png",
                    image_mode="RGBA",
                    height=IMG_HEIGHT,
                )
                mode = gr.Dropdown(
                    choices=list(MODES),
                    value="blocks",
                    label="Render mode",
                )
                with gr.Row():
                    num_colors = gr.Slider(
                        0, 64, value=16, step=1, label="Colors (0 = skip)"
                    )
                    initial_upscale = gr.Slider(
                        1, 4, value=2, step=1, label="Initial upscale"
                    )
                with gr.Row():
                    pixel_width = gr.Slider(
                        0, 50, value=0, step=1, label="Pixel width (0 = auto)"
                    )
                    edge_threshold = gr.Slider(
                        0.0,
                        0.5,
                        value=0.15,
                        step=0.01,
                        label="Edge threshold (ascii mode)",
                    )
                with gr.Row():
                    transparent = gr.Checkbox(value=False, label="Transparent bg")
                    ramp = gr.Textbox(
                        value="",
                        placeholder=" .:-=+*#%@",
                        label="Custom ASCII ramp",
                    )
                btn = gr.Button("Render", variant="primary")
            with gr.Column():
                html_out = gr.HTML(label="Preview")
                ansi_out = gr.Code(label="Raw ANSI", interactive=True)

        btn.click(
            fn=process,
            inputs=[
                input_img,
                mode,
                num_colors,
                transparent,
                initial_upscale,
                pixel_width,
                edge_threshold,
                ramp,
            ],
            outputs=[html_out, ansi_out],
        )

    return demo


def main():
    """Entry point for ``pta-web``."""
    demo = create_demo()
    demo.launch()


if __name__ == "__main__":
    main()
