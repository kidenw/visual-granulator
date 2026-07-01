from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from itertools import islice
import os
import tempfile

from PIL import Image
import streamlit as st

from granular import animated_webp_bytes, granulate_layers, granulate_video_frame, mp4_bytes, normalize_layers, preview_size


IMAGE_TYPES = ("png", "jpg", "jpeg", "jfif", "webp")
VIDEO_TYPES = ("mp4", "mov", "webm", "avi")


@st.cache_data(show_spinner=False)
def _video_frames(name: str, data: bytes, limit: int) -> list[Image.Image]:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError("Video upload needs imageio[ffmpeg]. Install requirements.txt again.") from error

    suffix = "." + name.rsplit(".", 1)[-1].lower()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
            file.write(data)
            temp_path = file.name
        reader = imageio.get_reader(temp_path)
        frames = [Image.fromarray(frame).convert("RGBA") for frame in islice(reader.iter_data(), limit)]
        return frames
    finally:
        try:
            reader.close()
        except UnboundLocalError:
            pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@st.cache_data(show_spinner=False)
def _media_bundle(files: tuple[tuple[str, bytes], ...], video_frame_limit: int) -> tuple[list[Image.Image], list[str], tuple[tuple[int, ...], ...]]:
    layers = []
    labels = []
    groups = []
    for name, data in files:
        ext = name.rsplit(".", 1)[-1].lower()
        if ext in VIDEO_TYPES:
            frames = _video_frames(name, data, video_frame_limit)
            if len(frames) > 1:
                groups.append(tuple(range(len(layers), len(layers) + len(frames))))
            layers.extend(frames)
            labels.extend(f"{name} #{index + 1}" for index in range(len(frames)))
        else:
            layers.append(Image.open(BytesIO(data)).convert("RGBA").copy())
            labels.append(name)
    return layers, labels, tuple(groups)


@st.cache_data(show_spinner=False)
def _normalized_layers(files: tuple[tuple[str, bytes], ...], size: tuple[int, int], video_frame_limit: int) -> list[Image.Image]:
    return normalize_layers(_media_bundle(files, video_frame_limit)[0], size)


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
        layers_are_normalized=True,
    )


def _video_frame(layers: list[Image.Image], grain_size: int, count: int, frame_index: int, frame_count: int) -> Image.Image:
    return granulate_video_frame(
        layers=layers,
        frame_index=frame_index,
        frame_count=frame_count,
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
        motion_amount=motion_amount,
        layer_switch=layer_switch,
        layer_switch_rate=layer_switch_rate / max(1, fps),
        size_animation=size_animation,
        sample_offset=sample_offset,
        pixel_stretch_jitter=pixel_stretch_jitter,
        edge_width_mode=edge_width_mode,
        video_layer_groups=video_layer_groups,
        video_time_dilation=video_time_dilation,
        layers_are_normalized=True,
    )


st.set_page_config(page_title="Visual Granulator", layout="wide")
st.title("Visual Granulator")

uploads = st.sidebar.file_uploader(
    "Images / Videos",
    type=IMAGE_TYPES + VIDEO_TYPES,
    accept_multiple_files=True,
)

if not uploads:
    st.info("Upload images to begin.")
    st.stop()

media_files = tuple((upload.name, upload.getvalue()) for upload in uploads)
video_frame_limit = st.sidebar.slider("Video Source Frames", 1, 720, 120)
import_progress = st.progress(0, text="Importing media")
import_progress.progress(5, text="Importing media")
images, layer_labels, video_layer_groups = _media_bundle(media_files, video_frame_limit)
import_progress.progress(70, text="Preparing layers")
import_progress.progress(100, text="Import complete")
import_progress.empty()
full_width = max(image.width for image in images)
full_height = max(image.height for image in images)
preview_width, preview_height, preview_scale = preview_size(full_width, full_height)

max_grain = max(2, min(full_width, full_height))
output_mode = st.sidebar.selectbox("Output Mode", ("Image", "Video"))
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

if output_mode == "Video":
    duration = st.sidebar.number_input("Duration", min_value=0.25, max_value=30.0, value=5.0, step=0.25)
    fps = st.sidebar.number_input("FPS", min_value=1, max_value=60, value=24, step=1)
    render_workers = st.sidebar.slider("CPU Render Workers", 1, max(1, min(8, os.cpu_count() or 1)), max(1, min(4, os.cpu_count() or 1)))
    video_export_format = st.sidebar.selectbox("Video Export", ("MP4", "Animated WebP"))
    motion_amount = st.sidebar.slider("Motion Amount", 0.0, 256.0, 24.0)
    sample_offset = st.sidebar.checkbox("Sample Offset", value=False)
    layer_switch = st.sidebar.checkbox("Layer Switch", value=False)
    layer_switch_rate = st.sidebar.slider("Layer Switch Rate", 0.0, 24.0, 2.0)
    size_animation = st.sidebar.checkbox("Size Animation", value=True)
    video_time_dilation = st.sidebar.slider("Video Time Dilation", 0.0, 1.0, 0.0)
    pixel_stretch_jitter = st.sidebar.slider("Pixel Stretch Jitter", 0.0, 1.0, 0.25)
    edge_width_mode = st.sidebar.selectbox("Edge Width Mode", ("None", "Static", "Varied"), index=1)
else:
    duration = 5.0
    fps = 24
    render_workers = 1
    video_export_format = "MP4"
    motion_amount = 0.0
    sample_offset = False
    layer_switch = False
    layer_switch_rate = 0.0
    size_animation = False
    video_time_dilation = 0.0
    pixel_stretch_jitter = 0.0
    edge_width_mode = "Static"

with st.sidebar.expander("Layer Weights"):
    layer_weights = [
        st.slider(label, 0.0, 100.0, 1.0, 0.1, key=f"weight-{index}-{label}")
        for index, label in enumerate(layer_labels)
    ]

outline_color = tuple(int(outline_rgb[i : i + 2], 16) for i in (1, 3, 5)) + (outline_alpha,)
preview_layers = _normalized_layers(media_files, (preview_width, preview_height), video_frame_limit)
preview_grain = max(1, round(grain_size * preview_scale))
frame_count = max(1, round(float(duration) * int(fps)))
preview = (
    _video_frame(preview_layers, preview_grain, min(grain_count, preview_density), frame_count // 3, frame_count)
    if output_mode == "Video"
    else _render(preview_layers, preview_grain, min(grain_count, preview_density))
)

left, right = st.columns(2)
left.image(preview_layers[0], caption=f"Layer stack preview ({len(images)} images)", use_container_width=True)
right.image(preview, caption=f"Preview {preview_width} x {preview_height}", use_container_width=True)

if output_mode == "Image" and st.button("Render Full PNG"):
    render_progress = st.progress(0, text="Rendering PNG")
    result = _render(_normalized_layers(media_files, (full_width, full_height), video_frame_limit), grain_size, grain_count)
    render_progress.progress(100, text="Render complete")
    buffer = BytesIO()
    result.save(buffer, format="PNG")
    render_progress.empty()
    st.download_button(
        "Download Full PNG",
        data=buffer.getvalue(),
        file_name="visual-granulator.png",
        mime="image/png",
    )

if output_mode == "Video" and st.button(f"Render Full {video_export_format}"):
    full_layers = _normalized_layers(media_files, (full_width, full_height), video_frame_limit)
    render_progress = st.progress(0, text="Rendering video")
    frames: list[Image.Image | None] = [None] * frame_count
    with ThreadPoolExecutor(max_workers=render_workers) as executor:
        futures = {
            executor.submit(_video_frame, full_layers, grain_size, grain_count, frame_index, frame_count): frame_index
            for frame_index in range(frame_count)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            frames[futures[future]] = future.result()
            render_progress.progress(round(done / frame_count * 100), text="Rendering video")
    complete_frames = [frame for frame in frames if frame is not None]
    render_progress.progress(100, text=f"Encoding {video_export_format}")
    data = mp4_bytes(complete_frames, int(fps)) if video_export_format == "MP4" else animated_webp_bytes(complete_frames, int(fps))
    render_progress.empty()
    st.download_button(
        f"Download {video_export_format}",
        data=data,
        file_name=f"visual-granulator.{('mp4' if video_export_format == 'MP4' else 'webp')}",
        mime=("video/mp4" if video_export_format == "MP4" else "image/webp"),
    )
