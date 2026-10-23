"""Общие фикстуры.

Вместо настоящей YOLO в тестах работает ``FakeModel``: тесты идут за
секунды, без видеокарты, без torch и без файла весов.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeBoxes:
    """Замена ``Results.boxes``: нужна только длина."""

    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count


class FakeResult:
    """Замена ``ultralytics.engine.results.Results``."""

    def __init__(
        self, count: int | None = 3, image: np.ndarray | None = None
    ) -> None:
        self.boxes = None if count is None else FakeBoxes(count)
        if image is None:
            # BGR: чисто синий цвет, в RGB он должен стать (0, 0, 255)
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[..., 0] = 255
        self._image = image

    def plot(self) -> np.ndarray:
        return self._image


class FakeModel:
    """Замена YOLO: запоминает вызовы и отдаёт заранее заданный результат."""

    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def predict(self, source: str, **kwargs: Any) -> list[FakeResult]:
        self.calls.append((source, kwargs))
        return [self.result]


def _make_image(
    path: Path,
    size: tuple[int, int] = (32, 24),
    mode: str = "RGB",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    color: Any = 128 if mode == "L" else (0, 128, 0)
    Image.new(mode, size, color).save(path)
    return path


@pytest.fixture
def make_image() -> Callable[..., Path]:
    """Фабрика тестовых картинок."""
    return _make_image


@pytest.fixture
def make_model() -> Callable[..., FakeModel]:
    """Фабрика фальшивых моделей с нужным числом найденных колосьев."""

    def factory(
        count: int | None = 3, image: np.ndarray | None = None
    ) -> FakeModel:
        return FakeModel(FakeResult(count=count, image=image))

    return factory


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """Папка с тремя картинками разных форматов и одним лишним файлом."""
    directory = tmp_path / "images"
    for name in ("b.png", "a.jpg", "c.JPEG"):
        _make_image(directory / name)
    (directory / "notes.txt").write_text("не картинка", encoding="utf-8")
    return directory


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """Датасет в формате YOLO: 10 фото, у последнего нет разметки."""
    data_dir = tmp_path / "data"
    labels_dir = data_dir / "train" / "labels"
    labels_dir.mkdir(parents=True)
    for index in range(10):
        _make_image(data_dir / "train" / "images" / f"img_{index:02d}.png")
        if index < 9:
            (labels_dir / f"img_{index:02d}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return data_dir


@pytest.fixture
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Временный корень проекта с копией configs/ и чистым окружением.

    ``os.environ`` подменяется копией без переменных WHEAT_*, поэтому
    то, что тест загрузит из ``.env``, не просочится в другие тесты.
    """
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("WHEAT_")
    }
    monkeypatch.setattr(os, "environ", clean_env)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def restore_package_logger() -> Iterator[None]:
    """Вернуть логгер пакета в исходное состояние после теста.

    ``setup_logging`` из CLI меняет обработчики и отключает передачу
    сообщений наверх, а ``caplog`` в других тестах это ломает.
    """
    package_logger = logging.getLogger("wheat_detection")
    saved = (
        package_logger.handlers[:],
        package_logger.level,
        package_logger.propagate,
    )
    yield
    package_logger.handlers[:] = saved[0]
    package_logger.setLevel(saved[1])
    package_logger.propagate = saved[2]
