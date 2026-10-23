"""Сохранение и показ картинок: результат с рамками и сравнение «до и после».

Рисунки строятся через объект ``Figure`` без ``pyplot``: так код работает
и на сервере без графического окна, и в тестах.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from PIL import Image

Image3 = NDArray[np.uint8]


def bgr_to_rgb(image: Image3) -> Image3:
    """Поменять порядок каналов BGR (OpenCV) на RGB."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Ожидалось цветное изображение HxWx3, получено {image.shape}"
        )
    return np.ascontiguousarray(image[..., ::-1])


def load_rgb(path: Path) -> Image3:
    """Прочитать изображение с диска в формате RGB."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_image(image: Image3, output_path: Path) -> Path:
    """Сохранить RGB-изображение, создав папку при необходимости."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return output_path


def build_comparison(
    original: Image3,
    annotated: Image3,
    image_name: str,
    count: int,
) -> Figure:
    """Собрать рисунок из двух частей: исходное фото и фото с рамками."""
    figure = Figure(figsize=(15, 7), layout="tight")
    before, after = figure.subplots(1, 2)

    before.imshow(original)
    before.set_title(f"ДО ОБРАБОТКИ\n{image_name}", fontsize=12)
    before.axis("off")

    after.imshow(annotated)
    after.set_title(f"ПОСЛЕ ОБРАБОТКИ\nНайдено колосьев: {count}", fontsize=12)
    after.axis("off")
    return figure


def save_figure(figure: Figure, output_path: Path, dpi: int = 100) -> Path:
    """Сохранить рисунок в файл, создав папку при необходимости."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    return output_path


def show_file(path: Path) -> None:
    """Открыть картинку в стандартной программе просмотра."""
    with Image.open(path) as image:
        image.show(title=path.name)
