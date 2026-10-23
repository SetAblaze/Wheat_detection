"""Поиск изображений и разбиение датасета на обучающую и проверочную части.

В исходной версии проекта в ``dataset.yaml`` проверочная выборка совпадала
с обучающей (``val: train/images``), поэтому метрики были завышены. Здесь
фото делятся на непересекающиеся train и val воспроизводимо (по ``seed``),
а для YOLO создаются списки файлов и описание датасета с абсолютными путями.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wheat_detection.config import ConfigError

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
TRAIN_LIST_NAME = "train.txt"
VAL_LIST_NAME = "val.txt"
DATASET_YAML_NAME = "dataset.yaml"


@dataclass(frozen=True)
class DatasetSplit:
    """Разбиение изображений на обучающую и проверочную части."""

    train: tuple[Path, ...]
    val: tuple[Path, ...]


@dataclass(frozen=True)
class SplitFiles:
    """Файлы, по которым YOLO находит обучающие и проверочные фото."""

    train_list: Path
    val_list: Path
    dataset_yaml: Path


def list_images(directory: Path) -> list[Path]:
    """Вернуть отсортированный список изображений в папке (без подпапок)."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Папка с изображениями не найдена: {directory}"
        )
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def pick_random_image(directory: Path, seed: int | None = None) -> Path:
    """Выбрать случайное изображение из папки.

    С одним и тем же ``seed`` выбор повторяется, это удобно для отладки.
    """
    images = list_images(directory)
    if not images:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise FileNotFoundError(
            f"В папке {directory} нет изображений ({extensions})"
        )
    return random.Random(seed).choice(images)


def label_path_for(image_path: Path) -> Path:
    """Путь к файлу разметки YOLO для изображения.

    Правило то же, что в ultralytics: последняя папка ``images`` в пути
    меняется на ``labels``, расширение на ``.txt``. Например,
    ``data/train/images/a.png`` даёт ``data/train/labels/a.txt``.
    """
    parts = list(image_path.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def find_images_without_labels(images: Sequence[Path]) -> list[Path]:
    """Найти изображения, для которых нет файла разметки."""
    return [image for image in images if not label_path_for(image).is_file()]


def split_images(
    images: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> DatasetSplit:
    """Случайно и воспроизводимо разбить изображения на train и val.

    В каждой части остаётся хотя бы одно изображение.
    """
    if not 0 < val_fraction < 1:
        raise ValueError(
            "Доля проверочных фото должна быть между 0 и 1, "
            f"а не {val_fraction}"
        )
    if len(images) < 2:
        raise ValueError("Для разбиения нужно хотя бы два изображения")

    shuffled = sorted(images)  # порядок на входе не влияет на результат
    random.Random(seed).shuffle(shuffled)
    val_size = round(len(shuffled) * val_fraction)
    val_size = min(max(val_size, 1), len(shuffled) - 1)
    return DatasetSplit(
        train=tuple(sorted(shuffled[val_size:])),
        val=tuple(sorted(shuffled[:val_size])),
    )


def split_files_in(splits_dir: Path) -> SplitFiles:
    """Пути к файлам разбиения внутри папки ``splits_dir``."""
    return SplitFiles(
        train_list=splits_dir / TRAIN_LIST_NAME,
        val_list=splits_dir / VAL_LIST_NAME,
        dataset_yaml=splits_dir / DATASET_YAML_NAME,
    )


def write_split(
    split: DatasetSplit,
    splits_dir: Path,
    dataset_template: Path,
) -> SplitFiles:
    """Записать списки train и val и описание датасета для YOLO.

    Пути в файлах абсолютные, поэтому YOLO не зависит от текущей папки.
    Файлы создаются на каждой машине заново и в git не хранятся.
    """
    template = read_dataset_template(dataset_template)
    files = split_files_in(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    _write_image_list(files.train_list, split.train)
    _write_image_list(files.val_list, split.val)

    dataset = {
        **template,
        "path": splits_dir.resolve().as_posix(),
        "train": files.train_list.resolve().as_posix(),
        "val": files.val_list.resolve().as_posix(),
    }
    with files.dataset_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dataset, file, allow_unicode=True, sort_keys=False)
    return files


def prepare_split(
    images_dir: Path,
    splits_dir: Path,
    dataset_template: Path,
    val_fraction: float,
    seed: int,
) -> SplitFiles:
    """Найти фото для обучения, разбить их и записать файлы для YOLO."""
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"В папке {images_dir} нет фото для обучения")

    unlabeled = find_images_without_labels(images)
    if unlabeled:
        logger.warning(
            "У %d из %d фото нет файла разметки, YOLO посчитает их фоном "
            "(пример: %s)",
            len(unlabeled),
            len(images),
            unlabeled[0],
        )

    split = split_images(images, val_fraction=val_fraction, seed=seed)
    files = write_split(split, splits_dir, dataset_template)
    logger.info(
        "Разбиение готово: train %d фото, val %d фото (seed=%d)",
        len(split.train),
        len(split.val),
        seed,
    )
    logger.info("Описание датасета для YOLO: %s", files.dataset_yaml)
    return files


def read_dataset_template(path: Path) -> dict[str, Any]:
    """Прочитать шаблон описания датасета (классы объектов)."""
    if not path.is_file():
        raise FileNotFoundError(f"Шаблон описания датасета не найден: {path}")
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict) or not data.get("names"):
        raise ConfigError(f"{path}: в описании датасета нет классов (names)")
    return data


def _write_image_list(path: Path, images: Sequence[Path]) -> None:
    lines = [image.resolve().as_posix() for image in images]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
