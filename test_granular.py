from __future__ import annotations

from PIL import Image
import numpy as np

from granular import _grain_plan, granulate_image, granulate_layers, normalize_layers, preview_size


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


if __name__ == "__main__":
    test_granulate_image_is_seeded_and_clamps_grain_size()
    test_layers_make_one_output_and_sample_multiple_z_layers()
    test_layer_weights_can_exclude_a_layer()
    test_grain_size_variation_changes_planned_sizes()
    test_grain_size_distributions_stay_in_low_high_bounds()
    test_pixel_stretch_fills_from_original_source_layer()
    test_normalize_layers_and_preview_size()
    test_style_changes_reuse_seeded_grain_plan()
