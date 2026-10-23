"""Загрузка модели и поиск колосьев на изображении."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wheat_detection.config import PredictConfig
from wheat_detection.visualization import Image3, bgr_to_rgb

logger = logging.getLogger(__name__)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


class DetectionModel(Protocol):
    """Всё, что нужно от модели: метод ``predict`` как у ultralytics.YOLO."""

    def predict(self, source: str, **kwargs: Any) -> Any:
        """Вернуть список результатов, по одному на изображение."""


@dataclass(frozen=True)
class Prediction:
    """Результат поиска колосьев на одном фото."""

    image_path: Path
    count: int
    annotated: Image3  # фото с рамками, каналы RGB


def resolve_device(device: str) -> str | None:
    """Перевести настройку устройства в формат ultralytics.

    ``auto`` превращается в ``None``: тогда ultralytics сам возьмёт
    видеокарту, а если её нет, процессор. В исходном коде было жёстко
    ``device=0``, и на компьютере без видеокарты обучение падало.
    """
    value = device.strip().lower()
    return None if value in {"", "auto"} else value


def is_lfs_pointer(path: Path) -> bool:
    """Проверить, что файл лишь ссылка Git LFS, а не сами данные."""
    with path.open("rb") as file:
        return file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def load_model(weights: Path) -> DetectionModel:
    """Загрузить модель YOLO из файла весов."""
    if not weights.is_file():
        raise FileNotFoundError(
            f"Файл весов не найден: {weights}. Обучите модель командой "
            "`wheat-detection train` или укажите путь через --weights"
        )
    if is_lfs_pointer(weights):
        raise FileNotFoundError(
            f"{weights} пока только ссылка Git LFS, сами веса не скачаны. "
            "Выполните: git lfs install && git lfs pull"
        )

    # ultralytics тянет за собой torch, поэтому импортируем только здесь:
    # команды и тесты, которым модель не нужна, работают быстро
    from ultralytics import YOLO

    logger.debug("Загружаю модель из %s", weights)
    model: DetectionModel = YOLO(str(weights))
    return model


def predict_image(
    model: DetectionModel,
    image_path: Path,
    params: PredictConfig,
) -> Any:
    """Запустить модель на одном фото и вернуть результат ultralytics."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    results = model.predict(
        source=str(image_path),
        conf=params.conf,
        iou=params.iou,
        imgsz=params.imgsz,
        device=resolve_device(params.device),
        max_det=params.max_det,
        verbose=False,
    )
    return results[0]


def count_detections(result: Any) -> int:
    """Сколько объектов (колосьев) нашла модель."""
    boxes = getattr(result, "boxes", None)
    return 0 if boxes is None else len(boxes)


def detect(
    model: DetectionModel,
    image_path: Path,
    params: PredictConfig,
) -> Prediction:
    """Найти колосья на фото и подготовить картинку с рамками.

    ``result.plot()`` в ultralytics отдаёт картинку в порядке каналов BGR.
    В исходном скрипте её показывали через matplotlib как есть, из-за чего
    синий и красный цвета менялись местами. Здесь каналы переставляются.
    """
    result = predict_image(model, image_path, params)
    return Prediction(
        image_path=image_path,
        count=count_detections(result),
        annotated=bgr_to_rgb(result.plot()),
    )
