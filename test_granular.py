from __future__ import annotations

from io import BytesIO

from PIL import Image
import numpy as np

from granular import (
    _grain_plan,
    animated_webp_bytes,
    granulate_image,
    granulate_layers,
    granulate_video_frame,
    mp4_bytes,
    normalize_layers,
    preview_size,
)


def _image(mode: str = "RGB") -> Image.Image:
    data = np.zeros((24, 32, 4), dtype=np.uint8)
    data[..., 0] = np.arange(32, dtype=np.uint8)
    data[..., 1] = np.arange(24, dtype=np.uint8)[:, None]
    data[..., 2] = 120
    data[..., 3] = 255
    return Image.fromarray(data, "RGBA").convert(mode)


def _render(image: Image.Image, seed: int) -> Image.Image:
    return granulate_image(
        image=image,
        grain_size=99,
        grain_count=80,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.5,
        spray=0.5,
        edge_fade=0.2,
        outline_width=1,
        outline_color=(255, 255, 255, 200),
        seed=seed,
    )


def test_granulate_image_is_seeded_and_clamps_grain_size() -> None:
    first = _render(_image("RGB"), 7)
    second = _render(_image("RGB"), 7)
    third = _render(_image("RGBA"), 8)

    assert first.size == (32, 24)
    assert first.mode == "RGBA"
    assert list(first.getdata()) == list(second.getdata())
    assert list(first.getdata()) != list(third.getdata())


def test_layers_make_one_output_and_sample_multiple_z_layers() -> None:
    red = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    blue = Image.new("RGBA", (16, 16), (0, 0, 255, 255))
    result = granulate_layers(
        layers=[red, blue],
        grain_size=1,
        grain_count=200,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.5,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=4,
    )
    colors = set(result.getdata())

    assert result.size == (16, 16)
    assert (255, 0, 0, 255) in colors
    assert (0, 0, 255, 255) in colors


def test_layer_weights_can_exclude_a_layer() -> None:
    red = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    blue = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
    result = granulate_layers(
        layers=[red, blue],
        grain_size=1,
        grain_count=64,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.5,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=4,
        layer_weights=[0, 1],
    )
    colors = set(result.getdata())

    assert (0, 0, 255, 255) in colors
    assert (255, 0, 0, 255) not in colors


def test_grain_size_variation_changes_planned_sizes() -> None:
    plan = _grain_plan(32, 32, 1, (1.0,), 8, 0.25, 3.0, "Uniform", 80, 0.5, 0.5, 0.5, 0, 3)
    sizes = {grain[-1] for grain in plan}

    assert len(sizes) > 1


def test_grain_size_distributions_stay_in_low_high_bounds() -> None:
    for mode in ("Uniform", "Normal", "Exponential"):
        plan = _grain_plan(32, 32, 1, (1.0,), 10, 0.1, 4.0, mode, 80, 0.5, 0.5, 0.5, 0, 3)
        sizes = [grain[-1] for grain in plan]

        assert min(sizes) >= 1
        assert max(sizes) <= 32


def test_pixel_stretch_fills_from_original_source_layer() -> None:
    source = Image.new("RGBA", (8, 8), (0, 255, 0, 255))
    result = granulate_layers(
        layers=[source],
        grain_size=2,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=1,
        outline_color=(255, 0, 0, 255),
        seed=1,
        background_mode="Pixel Stretch",
        stretch_axis="X",
    )

    assert result.getpixel((0, 0)) == (0, 255, 0, 255)
    assert (255, 0, 0, 255) in set(result.getdata())


def test_normalize_layers_and_preview_size() -> None:
    wide = Image.new("RGBA", (2000, 500), (255, 0, 0, 255))
    tall = Image.new("RGBA", (400, 1000), (0, 0, 255, 255))
    layers = normalize_layers([wide, tall])

    assert layers[0].size == (2000, 1000)
    assert layers[1].size == (2000, 1000)
    assert preview_size(2000, 1000) == (1024, 512, 0.512)


def test_style_changes_reuse_seeded_grain_plan() -> None:
    source = Image.new("RGBA", (12, 12), (0, 255, 0, 255))
    _grain_plan.cache_clear()

    granulate_layers(
        layers=[source],
        grain_size=3,
        grain_count=8,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.25,
        spray=0.1,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=2,
    )
    granulate_layers(
        layers=[source],
        grain_size=3,
        grain_count=8,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.25,
        spray=0.1,
        edge_fade=0.5,
        outline_width=1,
        outline_color=(255, 0, 0, 255),
        seed=2,
        background_mode="Color Fill",
    )

    assert _grain_plan.cache_info().hits == 1


def test_video_frames_change_with_motion_and_size_animation() -> None:
    layers = [_image("RGBA")]
    first = granulate_video_frame(
        layers=layers,
        frame_index=0,
        frame_count=12,
        grain_size=6,
        grain_count=30,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.5,
        spray=0.3,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=5,
        grain_size_low=0.5,
        grain_size_high=2.0,
        motion_amount=4,
        size_animation=True,
    )
    later = granulate_video_frame(
        layers=layers,
        frame_index=6,
        frame_count=12,
        grain_size=6,
        grain_count=30,
        source_x=0.5,
        source_y=0.5,
        source_spread=0.5,
        spray=0.3,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=5,
        grain_size_low=0.5,
        grain_size_high=2.0,
        motion_amount=4,
        size_animation=True,
    )

    assert list(first.getdata()) != list(later.getdata())


def test_video_size_animation_keeps_grain_center_stable() -> None:
    source = Image.new("RGBA", (32, 32), (0, 255, 0, 255))

    def center(frame: Image.Image) -> tuple[float, float]:
        alpha = np.asarray(frame.getchannel("A"))
        ys, xs = np.nonzero(alpha)
        return float(xs.mean()), float(ys.mean())

    first = granulate_video_frame(
        layers=[source],
        frame_index=0,
        frame_count=12,
        grain_size=10,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=5,
        motion_amount=0,
        size_animation=True,
    )
    later = granulate_video_frame(
        layers=[source],
        frame_index=8,
        frame_count=12,
        grain_size=10,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=5,
        motion_amount=0,
        size_animation=True,
    )

    ax, ay = center(first)
    bx, by = center(later)
    assert abs(ax - bx) <= 1
    assert abs(ay - by) <= 1


def test_video_sample_offset_can_be_disabled() -> None:
    data = np.zeros((32, 32, 4), dtype=np.uint8)
    data[..., 0] = np.arange(32, dtype=np.uint8)
    data[..., 3] = 255
    source = Image.fromarray(data, "RGBA")

    def colors(frame: Image.Image) -> list[tuple[int, int, int, int]]:
        return sorted(pixel for pixel in frame.getdata() if pixel[3])

    kwargs = dict(
        layers=[source],
        frame_count=12,
        grain_size=8,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=5,
        motion_amount=6,
        size_animation=False,
    )
    still_sample = colors(granulate_video_frame(**kwargs, frame_index=0, sample_offset=False))
    moved_grain = colors(granulate_video_frame(**kwargs, frame_index=8, sample_offset=False))
    offset_sample = colors(granulate_video_frame(**kwargs, frame_index=8, sample_offset=True))

    assert still_sample == moved_grain
    assert moved_grain != offset_sample


def test_video_layer_switch_can_show_multiple_layers() -> None:
    red = Image.new("RGBA", (12, 12), (255, 0, 0, 255))
    blue = Image.new("RGBA", (12, 12), (0, 0, 255, 255))
    colors = set()

    for frame_index in range(8):
        frame = granulate_video_frame(
            layers=[red, blue],
            frame_index=frame_index,
            frame_count=8,
            grain_size=2,
            grain_count=30,
            source_x=0.5,
            source_y=0.5,
            source_spread=0.5,
            spray=0,
            edge_fade=0,
            outline_width=0,
            outline_color=(255, 255, 255, 255),
            seed=4,
            layer_weights=[1, 1],
            layer_switch=True,
            layer_switch_rate=1,
        )
        colors.update(frame.getdata())

    assert (255, 0, 0, 255) in colors
    assert (0, 0, 255, 255) in colors


def test_video_layer_group_advances_one_sample_per_render_frame() -> None:
    layers = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 255, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 255)),
    ]
    kwargs = dict(
        layers=layers,
        frame_count=4,
        grain_size=8,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=2,
        layer_weights=[0, 1, 0],
        size_animation=False,
        video_layer_groups=((0, 1, 2),),
    )

    first = set(granulate_video_frame(**kwargs, frame_index=0).getdata())
    second = set(granulate_video_frame(**kwargs, frame_index=1).getdata())
    looped = set(granulate_video_frame(**kwargs, frame_index=3).getdata())

    assert first == {(0, 255, 0, 255)}
    assert second == {(0, 0, 255, 255)}
    assert looped == first


def test_video_time_dilation_offsets_video_grain_start() -> None:
    layers = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 255, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 255)),
    ]
    frame = granulate_video_frame(
        layers=layers,
        frame_index=0,
        frame_count=4,
        grain_size=8,
        grain_count=1,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=0,
        outline_color=(255, 255, 255, 255),
        seed=2,
        layer_weights=[0, 1, 0],
        size_animation=False,
        video_layer_groups=((0, 1, 2),),
        video_time_dilation=1,
    )

    assert set(frame.getdata()) != {(0, 255, 0, 255)}


def test_video_edge_width_modes() -> None:
    source = Image.new("RGBA", (16, 16), (0, 255, 0, 255))
    kwargs = dict(
        layers=[source],
        frame_index=2,
        frame_count=8,
        grain_size=8,
        grain_count=4,
        source_x=0.5,
        source_y=0.5,
        source_spread=0,
        spray=0,
        edge_fade=0,
        outline_width=2,
        outline_color=(255, 0, 0, 255),
        seed=1,
        motion_amount=0,
    )

    none_colors = set(granulate_video_frame(**kwargs, edge_width_mode="None").getdata())
    static_colors = set(granulate_video_frame(**kwargs, edge_width_mode="Static").getdata())
    varied_colors = set(granulate_video_frame(**kwargs, edge_width_mode="Varied").getdata())

    assert (255, 0, 0, 255) not in none_colors
    assert (255, 0, 0, 255) in static_colors
    assert (255, 0, 0, 255) in varied_colors


def test_animated_webp_bytes_round_trip() -> None:
    frames = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 255)),
    ]
    data = animated_webp_bytes(frames, 12)
    image = Image.open(BytesIO(data))

    assert image.format == "WEBP"
    assert getattr(image, "is_animated", False)


def test_mp4_bytes_writes_file_data() -> None:
    frames = [
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
        Image.new("RGBA", (8, 8), (0, 0, 255, 255)),
    ]
    data = mp4_bytes(frames, 12)

    assert b"ftyp" in data[:32]


if __name__ == "__main__":
    test_granulate_image_is_seeded_and_clamps_grain_size()
    test_layers_make_one_output_and_sample_multiple_z_layers()
    test_layer_weights_can_exclude_a_layer()
    test_grain_size_variation_changes_planned_sizes()
    test_grain_size_distributions_stay_in_low_high_bounds()
    test_pixel_stretch_fills_from_original_source_layer()
    test_normalize_layers_and_preview_size()
    test_style_changes_reuse_seeded_grain_plan()
    test_video_frames_change_with_motion_and_size_animation()
    test_video_size_animation_keeps_grain_center_stable()
    test_video_sample_offset_can_be_disabled()
    test_video_layer_switch_can_show_multiple_layers()
    test_video_layer_group_advances_one_sample_per_render_frame()
    test_video_time_dilation_offsets_video_grain_start()
    test_video_edge_width_modes()
    test_animated_webp_bytes_round_trip()
    test_mp4_bytes_writes_file_data()
