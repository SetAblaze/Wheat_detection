"""Обучение модели YOLO на датасете колосьев."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from wheat_detection.config import Settings
from wheat_detection.data import prepare_split, split_files_in
from wheat_detection.inference import resolve_device

logger = logging.getLogger(__name__)


def train(settings: Settings, run_name: str | None = None) -> Path:
    """Обучить модель и вернуть путь к лучшим весам.

    Если разбиение на train и val ещё не создано, оно делается
    автоматически с параметрами из настроек. Результаты обучения
    складываются в ``<runs_dir>/train/<run_name>``.
    """
    paths, params = settings.paths, settings.train

    files = split_files_in(paths.splits_dir)
    if not files.dataset_yaml.is_file():
        logger.info("Разбиение данных не найдено, создаю его")
        files = prepare_split(
            images_dir=paths.train_images_dir,
            splits_dir=paths.splits_dir,
            dataset_template=paths.dataset_template,
            val_fraction=params.val_fraction,
            seed=params.seed,
        )

    from ultralytics import YOLO  # тяжёлый импорт, см. inference.load_model

    name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(
        "Обучение: %s, эпох %d, batch %d, imgsz %d, устройство %s",
        params.base_model,
        params.epochs,
        params.batch,
        params.imgsz,
        params.device,
    )
    model = YOLO(params.base_model)
    model.train(
        data=str(files.dataset_yaml),
        epochs=params.epochs,
        imgsz=params.imgsz,
        batch=params.batch,
        device=resolve_device(params.device),
        workers=params.workers,
        seed=params.seed,
        deterministic=True,
        project=str((paths.runs_dir / "train").resolve()),
        name=name,
        exist_ok=False,
    )

    best = Path(model.trainer.best)
    logger.info("Обучение завершено, лучшие веса: %s", best)
    logger.info(
        "Чтобы предсказания шли на новых весах, скопируйте файл в %s "
        "или передайте путь через --weights",
        paths.weights,
    )
    return best
