from __future__ import annotations

from functools import lru_cache

from io import BytesIO
import os
import tempfile

from PIL import Image, ImageDraw
import numpy as np


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def preview_size(width: int, height: int, max_width: int = 1024) -> tuple[int, int, float]:
    if width <= max_width:
        return width, height, 1.0

    scale = max_width / width
    return max_width, max(1, round(height * scale)), scale


def normalize_layers(images: list[Image.Image], canvas_size: tuple[int, int] | None = None) -> list[Image.Image]:
    layers = [image.convert("RGBA") for image in images]
    if not layers:
        raise ValueError("at least one image is required")

    width, height = canvas_size or (
        max(image.width for image in layers),
        max(image.height for image in layers),
    )
    normalized = []
    for image in layers:
        scale = min(width / image.width, height / image.height)
        layer_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer = image if image.size == layer_size else image.resize(layer_size, Image.Resampling.LANCZOS)
        canvas.alpha_composite(layer, ((width - layer.width) // 2, (height - layer.height) // 2))
        normalized.append(canvas)
    return normalized


def _edge_mask(width: int, height: int, fade: float) -> Image.Image:
    if fade <= 0:
        return Image.new("L", (width, height), 255)

    border = max(1, int(min(width, height) * min(fade, 1.0) * 0.5))
    y, x = np.ogrid[:height, :width]
    distance = np.minimum(np.minimum(x, width - 1 - x), np.minimum(y, height - 1 - y))
    alpha = (np.clip(distance / border, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(alpha, "L")


@lru_cache(maxsize=128)
def _grain_plan(
    width: int,
    height: int,
    layer_count: int,
    layer_weights: tuple[float, ...],
    size: int,
    size_low: float,
    size_high: float,
    size_distribution: str,
    count: int,
    source_x: float,
    source_y: float,
    source_spread: float,
    spray: float,
    seed: int,
) -> tuple[tuple[int, int, int, int, int, int], ...]:
    rng = np.random.default_rng(seed)
    weights = np.array(layer_weights[:layer_count], dtype=float)
    weights = np.where(weights > 0, weights, 0)
    weights = None if weights.sum() <= 0 else weights / weights.sum()
    base_x = np.clip(source_x, 0, 1) * (width - 1)
    base_y = np.clip(source_y, 0, 1) * (height - 1)
    spread_x = max(0.0, source_spread) * width
    spread_y = max(0.0, source_spread) * height
    spray_x = max(0.0, spray) * width
    spray_y = max(0.0, spray) * height
    grains = []
    low_size = _clamp(round(size * min(size_low, size_high)), 1, min(width, height))
    high_size = _clamp(round(size * max(size_low, size_high)), low_size, min(width, height))

    for _ in range(count):
        grain_size = size
        if low_size != high_size:
            if size_distribution == "Normal":
                t = float(np.clip(rng.normal(0.5, 0.18), 0, 1))
            elif size_distribution == "Exponential":
                t = float(np.clip(rng.exponential(0.35), 0, 1))
            else:
                t = float(rng.random())
            grain_size = _clamp(round(low_size + (high_size - low_size) * t), 1, min(width, height))

        layer_index = int(rng.choice(layer_count, p=weights))
        cx = base_x + rng.uniform(-spread_x, spread_x)
        cy = base_y + rng.uniform(-spread_y, spread_y)
        left = _clamp(round(cx - grain_size / 2), 0, width - grain_size)
        top = _clamp(round(cy - grain_size / 2), 0, height - grain_size)
        dx = _clamp(round(left + rng.uniform(-spray_x, spray_x)), 0, width - grain_size)
        dy = _clamp(round(top + rng.uniform(-spray_y, spray_y)), 0, height - grain_size)
        grains.append((layer_index, left, top, dx, dy, grain_size))

    return tuple(grains)


def _fill_from_sources(
    output: Image.Image,
    layers: list[Image.Image],
    layer_at: np.ndarray,
    source_x_at: np.ndarray,
    source_y_at: np.ndarray,
    axis: str,
    rng: np.random.Generator,
) -> Image.Image:
    data = np.asarray(output).copy()
    filled = layer_at >= 0
    height, width = filled.shape
    layer_data = np.stack([np.asarray(layer) for layer in layers])

    def take(y: int, x: int, first_axis: str) -> tuple[int, int, int] | None:
        if first_axis == "X":
            xs = np.flatnonzero(filled[y])
            if len(xs):
                x = int(xs[np.abs(xs - x).argmin()])
                return int(layer_at[y, x]), int(source_y_at[y, x]), int(source_x_at[y, x])
            ys = np.flatnonzero(filled.any(axis=1))
            if len(ys):
                y = int(ys[np.abs(ys - y).argmin()])
                xs = np.flatnonzero(filled[y])
                x = int(xs[np.abs(xs - x).argmin()])
                return int(layer_at[y, x]), int(source_y_at[y, x]), int(source_x_at[y, x])
        else:
            ys = np.flatnonzero(filled[:, x])
            if len(ys):
                y = int(ys[np.abs(ys - y).argmin()])
                return int(layer_at[y, x]), int(source_y_at[y, x]), int(source_x_at[y, x])
            xs = np.flatnonzero(filled.any(axis=0))
            if len(xs):
                x = int(xs[np.abs(xs - x).argmin()])
                ys = np.flatnonzero(filled[:, x])
                y = int(ys[np.abs(ys - y).argmin()])
                return int(layer_at[y, x]), int(source_y_at[y, x]), int(source_x_at[y, x])
        return None

    for y, x in np.argwhere(data[..., 3] == 0):
        first_axis = rng.choice(("X", "Y")) if axis == "Random" else axis
        source = take(int(y), int(x), str(first_axis))
        if source is not None:
            z, sy, sx = source
            data[y, x] = layer_data[z, sy, sx]

    return Image.fromarray(data.astype(np.uint8), "RGBA")


def _weights(layer_weights: list[float] | tuple[float, ...] | None, layer_count: int) -> tuple[float, ...]:
    weights = [float(weight) for weight in (layer_weights or [])]
    if len(weights) < layer_count:
        weights.extend([1.0] * (layer_count - len(weights)))
    return tuple(weights[:layer_count])


def _pick_layer(layer_count: int, weights: tuple[float, ...], seed: int) -> int:
    rng = np.random.default_rng(seed)
    weight_array = np.array(weights[:layer_count], dtype=float)
    weight_array = np.where(weight_array > 0, weight_array, 0)
    weight_array = None if weight_array.sum() <= 0 else weight_array / weight_array.sum()
    return int(rng.choice(layer_count, p=weight_array))


def _draw_plan(
    sources: list[Image.Image],
    plan: tuple[tuple[int, int, int, int, int, int], ...],
    edge_fade: float,
    outline_width: int,
    outline_color: tuple[int, int, int, int],
    background_mode: str,
    stretch_axis: str,
    fill_color: tuple[int, int, int, int],
    seed: int,
    frame_index: int = 0,
    frame_count: int = 1,
    motion_amount: float = 0.0,
    layer_switch: bool = False,
    layer_switch_rate: float = 1.0,
    size_animation: bool = False,
    sample_offset: bool = False,
    pixel_stretch_jitter: float = 0.0,
    edge_width_mode: str = "Static",
    layer_weights: tuple[float, ...] | None = None,
    video_layer_groups: tuple[tuple[int, ...], ...] = (),
    video_time_dilation: float = 0.0,
) -> Image.Image:
    width, height = sources[0].size
    output = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fade_masks: dict[int, Image.Image] = {}
    layer_at = np.full((height, width), -1, dtype=np.int16)
    source_x_at = np.zeros((height, width), dtype=np.int32)
    source_y_at = np.zeros((height, width), dtype=np.int32)
    layer_weights = layer_weights or (1.0,) * len(sources)
    video_layer_at = {layer: group for group in video_layer_groups for layer in group}

    for index, (layer_index, left, top, dx, dy, grain_size) in enumerate(plan):
        phase = (seed * 0.013 + index * 1.618 + frame_index * 0.071) % (2 * np.pi)
        source_cx = left + grain_size / 2
        source_cy = top + grain_size / 2
        dest_cx = dx + grain_size / 2
        dest_cy = dy + grain_size / 2
        if layer_switch:
            switch_step = max(0, int(frame_index * max(0.0, layer_switch_rate)))
            if switch_step:
                layer_index = _pick_layer(len(sources), layer_weights, seed + index * 10007 + switch_step)

        video_group = video_layer_at.get(layer_index)
        if video_group:
            offset = 0
            dilation = float(np.clip(video_time_dilation, 0, 1))
            if dilation:
                offset = int(np.random.default_rng(seed + index * 10007).choice((-1, 0, 1), p=(dilation / 2, 1 - dilation, dilation / 2)))
            layer_index = video_group[(video_group.index(layer_index) + frame_index + offset) % len(video_group)]

        if size_animation:
            grain_size = _clamp(round(grain_size * (0.75 + 0.25 * (1 + np.sin(phase)))), 1, min(width, height))

        if motion_amount:
            dest_cx += np.sin(phase) * motion_amount
            dest_cy += np.cos(phase * 1.31) * motion_amount
            if sample_offset:
                source_cx += np.sin(phase * 0.73) * motion_amount
                source_cy += np.cos(phase) * motion_amount

        dx = _clamp(round(dest_cx - grain_size / 2), 0, width - grain_size)
        dy = _clamp(round(dest_cy - grain_size / 2), 0, height - grain_size)
        left = _clamp(round(source_cx - grain_size / 2), 0, width - grain_size)
        top = _clamp(round(source_cy - grain_size / 2), 0, height - grain_size)

        grain_outline_width = outline_width
        if edge_width_mode == "None":
            grain_outline_width = 0
        elif edge_width_mode == "Varied":
            grain_outline_width = _clamp(round(outline_width * (0.5 + 0.5 * np.sin(phase))), 0, outline_width)

        source = sources[layer_index]
        fade_mask = fade_masks.setdefault(grain_size, _edge_mask(grain_size, grain_size, edge_fade))
        grain = source.crop((left, top, left + grain_size, top + grain_size))
        alpha = Image.composite(fade_mask, Image.new("L", grain.size, 0), grain.getchannel("A"))
        grain.putalpha(alpha)

        if grain_outline_width > 0:
            draw = ImageDraw.Draw(grain)
            for inset in range(min(grain_outline_width, grain_size // 2)):
                draw.rectangle(
                    (inset, inset, grain_size - 1 - inset, grain_size - 1 - inset),
                    outline=outline_color,
                )

        output.alpha_composite(grain, (dx, dy))
        mask = np.asarray(grain.getchannel("A")) > 0
        ys, xs = np.nonzero(mask)
        layer_at[dy + ys, dx + xs] = layer_index
        source_x_at[dy + ys, dx + xs] = left + xs
        source_y_at[dy + ys, dx + xs] = top + ys

    if background_mode == "Original Layer":
        background = sources[0].copy()
        background.alpha_composite(output)
        return background
    if background_mode == "Pixel Stretch":
        axis = stretch_axis
        if pixel_stretch_jitter and np.random.default_rng(seed + frame_index).random() < pixel_stretch_jitter:
            axis = "Random"
        return _fill_from_sources(
            output,
            sources,
            layer_at,
            source_x_at,
            source_y_at,
            axis,
            np.random.default_rng(seed + frame_index + 1),
        )
    if background_mode == "Color Fill":
        background = Image.new("RGBA", (width, height), fill_color)
        background.alpha_composite(output)
        return background

    return output


def granulate_layers(
    layers: list[Image.Image],
    grain_size: int,
    grain_count: int,
    source_x: float,
    source_y: float,
    source_spread: float,
    spray: float,
    edge_fade: float,
    outline_width: int,
    outline_color: tuple[int, int, int, int],
    seed: int,
    layer_weights: list[float] | tuple[float, ...] | None = None,
    grain_size_variation: float = 0.0,
    grain_size_low: float | None = None,
    grain_size_high: float | None = None,
    grain_size_distribution: str = "Uniform",
    background_mode: str = "Transparent",
    stretch_axis: str = "X",
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    layers_are_normalized: bool = False,
) -> Image.Image:
    sources = layers if layers_are_normalized else normalize_layers(layers)
    width, height = sources[0].size
    size = _clamp(int(grain_size), 1, min(width, height))
    count = max(0, int(grain_count))
    weights = _weights(layer_weights, len(sources))
    variation = float(np.clip(grain_size_variation, 0, 1))
    size_low = float(grain_size_low if grain_size_low is not None else 1 - variation)
    size_high = float(grain_size_high if grain_size_high is not None else 1 + variation)

    plan = _grain_plan(
        width,
        height,
        len(sources),
        weights,
        size,
        size_low,
        size_high,
        grain_size_distribution,
        count,
        source_x,
        source_y,
        source_spread,
        spray,
        seed,
    )

    return _draw_plan(
        sources,
        plan,
        edge_fade,
        outline_width,
        outline_color,
        background_mode,
        stretch_axis,
        fill_color,
        seed,
        layer_weights=weights,
    )


def granulate_video_frame(
    layers: list[Image.Image],
    frame_index: int,
    frame_count: int,
    grain_size: int,
    grain_count: int,
    source_x: float,
    source_y: float,
    source_spread: float,
    spray: float,
    edge_fade: float,
    outline_width: int,
    outline_color: tuple[int, int, int, int],
    seed: int,
    layer_weights: list[float] | tuple[float, ...] | None = None,
    grain_size_low: float | None = None,
    grain_size_high: float | None = None,
    grain_size_distribution: str = "Uniform",
    background_mode: str = "Transparent",
    stretch_axis: str = "X",
    fill_color: tuple[int, int, int, int] = (0, 0, 0, 255),
    motion_amount: float = 0.0,
    layer_switch: bool = True,
    layer_switch_rate: float = 1.0,
    size_animation: bool = True,
    sample_offset: bool = False,
    pixel_stretch_jitter: float = 0.0,
    edge_width_mode: str = "Static",
    video_layer_groups: tuple[tuple[int, ...], ...] = (),
    video_time_dilation: float = 0.0,
    layers_are_normalized: bool = False,
) -> Image.Image:
    sources = layers if layers_are_normalized else normalize_layers(layers)
    width, height = sources[0].size
    size = _clamp(int(grain_size), 1, min(width, height))
    weights = _weights(layer_weights, len(sources))
    size_low = float(1.0 if grain_size_low is None else grain_size_low)
    size_high = float(1.0 if grain_size_high is None else grain_size_high)
    plan = _grain_plan(
        width,
        height,
        len(sources),
        weights,
        size,
        size_low,
        size_high,
        grain_size_distribution,
        max(0, int(grain_count)),
        source_x,
        source_y,
        source_spread,
        spray,
        seed,
    )
    frame_t = 0 if frame_count <= 1 else frame_index / (frame_count - 1)
    return _draw_plan(
        sources,
        plan,
        edge_fade,
        outline_width,
        outline_color,
        background_mode,
        stretch_axis,
        fill_color,
        seed,
        frame_index=frame_index,
        frame_count=frame_count,
        motion_amount=motion_amount,
        layer_switch=layer_switch,
        layer_switch_rate=layer_switch_rate,
        size_animation=size_animation,
        sample_offset=sample_offset,
        pixel_stretch_jitter=pixel_stretch_jitter * (0.5 + 0.5 * np.sin(frame_t * 2 * np.pi)),
        edge_width_mode=edge_width_mode,
        layer_weights=weights,
        video_layer_groups=video_layer_groups,
        video_time_dilation=video_time_dilation,
    )


def animated_webp_bytes(frames: list[Image.Image], fps: int) -> bytes:
    if not frames:
        raise ValueError("at least one frame is required")
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=max(1, round(1000 / max(1, fps))),
        loop=0,
    )
    return buffer.getvalue()


def mp4_bytes(frames: list[Image.Image], fps: int) -> bytes:
    if not frames:
        raise ValueError("at least one frame is required")
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise RuntimeError("MP4 export needs imageio[ffmpeg]. Install requirements.txt again.") from error

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as file:
            temp_path = file.name
        writer = imageio.get_writer(temp_path, fps=max(1, int(fps)), codec="libx264", macro_block_size=1)
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
        writer.close()
        with open(temp_path, "rb") as file:
            return file.read()
    finally:
        try:
            writer.close()
        except UnboundLocalError:
            pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def granulate_image(image: Image.Image, **kwargs: object) -> Image.Image:
    return granulate_layers([image], **kwargs)
