from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wheat_detection.visualization import (
    bgr_to_rgb,
    build_comparison,
    load_rgb,
    save_figure,
    save_image,
)


def test_bgr_to_rgb_swaps_channels() -> None:
    image = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert bgr_to_rgb(image).tolist() == [[[3, 2, 1]]]


def test_bgr_to_rgb_rejects_grayscale() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        bgr_to_rgb(np.zeros((4, 4), dtype=np.uint8))


def test_load_rgb_converts_grayscale(
    tmp_path: Path, make_image: Callable[..., Path]
) -> None:
    path = make_image(tmp_path / "gray.png", size=(5, 4), mode="L")
    assert load_rgb(path).shape == (4, 5, 3)


def test_save_image_creates_directories(tmp_path: Path) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    output = save_image(image, tmp_path / "a" / "b" / "result.jpg")
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.size == (6, 4)


def test_comparison_figure(tmp_path: Path) -> None:
    original = np.zeros((10, 20, 3), dtype=np.uint8)
    annotated = np.full((10, 20, 3), 255, dtype=np.uint8)

    figure = build_comparison(original, annotated, "photo.png", count=7)
    titles = [axes.get_title() for axes in figure.axes]
    assert titles[0] == "ДО ОБРАБОТКИ\nphoto.png"
    assert titles[1] == "ПОСЛЕ ОБРАБОТКИ\nНайдено колосьев: 7"

    output = save_figure(figure, tmp_path / "out" / "compare.png", dpi=50)
    with Image.open(output) as saved:
        assert saved.size == (750, 350)
