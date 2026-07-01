from __future__ import annotations

from io import BytesIO

from PIL import Image
import streamlit as st

from granular import granulate_layers, normalize_layers, preview_size


@st.cache_data(show_spinner=False)
def _load_images(files: tuple[bytes, ...]) -> list[Image.Image]:
    return [Image.open(BytesIO(file)).convert("RGBA").copy() for file in files]


@st.cache_data(show_spinner=False)
def _normalized_layers(files: tuple[bytes, ...], size: tuple[int, int]) -> list[Image.Image]:
    return normalize_layers(_load_images(files), size)


def _render(layers: list[Image.Image], grain_size: int, count: int) -> Image.Image:
    return granulate_layers(
        layers=layers,
        grain_size=grain_size,
        grain_count=count,
        source_x=source_x,
        source_y=source_y,
        source_spread=source_spread,
        spray=spray,
        edge_fade=edge_fade,
        outline_width=outline_width,
        outline_color=outline_color,
        seed=int(seed),
        layer_weights=layer_weights,
        grain_size_low=grain_size_low,
        grain_size_high=grain_size_high,
        grain_size_distribution=grain_size_distribution,
        background_mode=background_mode,
        stretch_axis=stretch_axis,
        fill_color=fill_color,
    )


st.set_page_config(page_title="Visual Granulator", layout="wide")
st.title("Visual Granulator")

uploads = st.sidebar.file_uploader(
    "Images",
    type=("png", "jpg", "jpeg", "jfif", "webp"),
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload images to begin.")
    st.stop()

upload_bytes = tuple(upload.getvalue() for upload in uploads)
images = _load_images(upload_bytes)
full_width = max(image.width for image in images)
full_height = max(image.height for image in images)
preview_width, preview_height, preview_scale = preview_size(full_width, full_height)

max_grain = max(2, min(full_width, full_height))
grain_size = st.sidebar.slider("Grain Size", 2, max_grain, min(48, max_grain))
grain_size_distribution = st.sidebar.selectbox("Grain Size Mode", ("Uniform", "Normal", "Exponential"))
grain_size_low = st.sidebar.slider("Grain Size Low", 0.05, 4.0, 1.0)
grain_size_high = st.sidebar.slider("Grain Size High", 0.05, 8.0, 1.0)
grain_count = st.sidebar.slider("Density", 1, 1024, 800)
preview_density = st.sidebar.slider("Preview Density Cap", 64, 1024, 256)
source_x = st.sidebar.slider("Position X", 0.0, 1.0, 0.5)
source_y = st.sidebar.slider("Position Y", 0.0, 1.0, 0.5)
source_spread = st.sidebar.slider("Spread", 0.0, 1.0, 0.35)
spray = st.sidebar.slider("Spray", 0.0, 1.0, 0.2)
edge_fade = st.sidebar.slider("Edge Fade", 0.0, 1.0, 0.25)
outline_width = st.sidebar.slider("Outline Width", 0, 12, 0)
outline_rgb = st.sidebar.color_picker("Outline Color", "#ffffff")
outline_alpha = st.sidebar.slider("Outline Alpha", 0, 255, 180)
background_mode = st.sidebar.selectbox(
    "Unfilled Zones",
    ("Transparent", "Pixel Stretch", "Original Layer", "Color Fill"),
)
stretch_axis = "X"
if background_mode == "Pixel Stretch":
    stretch_axis = st.sidebar.selectbox("Stretch Axis", ("X", "Y", "Random"))
fill_color = (0, 0, 0, 255)
if background_mode == "Color Fill":
    fill_preset = st.sidebar.selectbox("Fill Color", ("Black", "White", "Custom"))
    fill_rgb = {
        "Black": "#000000",
        "White": "#ffffff",
    }.get(fill_preset, st.sidebar.color_picker("Custom Fill", "#000000"))
    fill_color = tuple(int(fill_rgb[i : i + 2], 16) for i in (1, 3, 5)) + (255,)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=999_999, value=1, step=1)

with st.sidebar.expander("Image Weights"):
    layer_weights = [
        st.slider(upload.name, 0.0, 100.0, 1.0, 0.1, key=f"weight-{index}")
        for index, upload in enumerate(uploads)
    ]

outline_color = tuple(int(outline_rgb[i : i + 2], 16) for i in (1, 3, 5)) + (outline_alpha,)
preview_layers = _normalized_layers(upload_bytes, (preview_width, preview_height))
preview_grain = max(1, round(grain_size * preview_scale))
preview = _render(preview_layers, preview_grain, min(grain_count, preview_density))

left, right = st.columns(2)
left.image(preview_layers[0], caption=f"Layer stack preview ({len(images)} images)", use_container_width=True)
right.image(preview, caption=f"Preview {preview_width} x {preview_height}", use_container_width=True)

if st.button("Render Full PNG"):
    result = _render(_normalized_layers(upload_bytes, (full_width, full_height)), grain_size, grain_count)
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    st.download_button(
        "Download Full PNG",
        data=buffer.getvalue(),
        file_name="visual-granulator.png",
        mime="image/png",
    )
