"""Проверка на настоящей модели.

Нужны установленный ultralytics и скачанные веса ``models/best.pt``
(через Git LFS). Если чего-то нет, тесты пропускаются, а не падают:
так они не мешают в CI, где веса не скачиваются.
Запустить только их: ``poetry run pytest -m slow``.
"""

from pathlib import Path
from typing import Any

import pytest

from wheat_detection.cli import main
from wheat_detection.config import PredictConfig
from wheat_detection.inference import detect, is_lfs_pointer, load_model

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "models" / "best.pt"
SAMPLE = REPO_ROOT / "tests" / "data" / "wheat_sample.jpg"


@pytest.fixture(scope="module")
def model() -> Any:
    # Сначала веса: в CI их нет, и ultralytics там даже не импортируется
    if not WEIGHTS.is_file() or is_lfs_pointer(WEIGHTS):
        pytest.skip("нет весов models/best.pt (выполните git lfs pull)")
    # ImportError, а не только «модуль не найден»: на сервере без
    # графических библиотек (нет libGL) падает импорт opencv
    pytest.importorskip("ultralytics", exc_type=ImportError)
    return load_model(WEIGHTS)


def test_real_model_finds_wheat_heads(model: Any) -> None:
    prediction = detect(model, SAMPLE, PredictConfig(device="cpu"))
    assert prediction.count > 0
    assert prediction.annotated.ndim == 3


def test_compare_command_on_real_model(model: Any, project_dir: Path) -> None:
    args = [
        "compare",
        str(SAMPLE),
        "--weights",
        str(WEIGHTS),
        "--device",
        "cpu",
        "--output-dir",
        "out",
    ]
    assert main(args) == 0
    assert (project_dir / "out" / "wheat_sample_compare.png").is_file()
