from __future__ import annotations

from functools import lru_cache

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
) -> Image.Image:
    sources = normalize_layers(layers)
    width, height = sources[0].size
    size = _clamp(int(grain_size), 1, min(width, height))
    count = max(0, int(grain_count))
    weights = tuple(float(weight) for weight in (layer_weights or [1.0] * len(sources)))
    variation = float(np.clip(grain_size_variation, 0, 1))
    size_low = float(grain_size_low if grain_size_low is not None else 1 - variation)
    size_high = float(grain_size_high if grain_size_high is not None else 1 + variation)

    output = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    fade_masks: dict[int, Image.Image] = {}
    layer_at = np.full((height, width), -1, dtype=np.int16)
    source_x_at = np.zeros((height, width), dtype=np.int32)
    source_y_at = np.zeros((height, width), dtype=np.int32)
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

    for layer_index, left, top, dx, dy, grain_size in plan:
        source = sources[layer_index]
        fade_mask = fade_masks.setdefault(grain_size, _edge_mask(grain_size, grain_size, edge_fade))
        grain = source.crop((left, top, left + grain_size, top + grain_size))
        alpha = Image.composite(fade_mask, Image.new("L", grain.size, 0), grain.getchannel("A"))
        grain.putalpha(alpha)

        if outline_width > 0:
            draw = ImageDraw.Draw(grain)
            for inset in range(min(outline_width, grain_size // 2)):
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
        return _fill_from_sources(
            output,
            sources,
            layer_at,
            source_x_at,
            source_y_at,
            stretch_axis,
            np.random.default_rng(seed + 1),
        )
    if background_mode == "Color Fill":
        background = Image.new("RGBA", (width, height), fill_color)
        background.alpha_composite(output)
        return background

    return output


def granulate_image(image: Image.Image, **kwargs: object) -> Image.Image:
    return granulate_layers([image], **kwargs)
