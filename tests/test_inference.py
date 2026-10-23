from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from wheat_detection.config import PredictConfig
from wheat_detection.inference import (
    count_detections,
    detect,
    is_lfs_pointer,
    load_model,
    predict_image,
    resolve_device,
)

LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:0000\n"
    "size 6250730\n"
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", None), (" AUTO ", None), ("", None), ("cpu", "cpu"), ("0", "0")],
)
def test_resolve_device(value: str, expected: str | None) -> None:
    assert resolve_device(value) == expected


@pytest.mark.parametrize(("count", "expected"), [(5, 5), (0, 0), (None, 0)])
def test_count_detections(
    make_model: Callable[..., Any], count: int | None, expected: int
) -> None:
    assert count_detections(make_model(count=count).result) == expected


def test_count_detections_without_boxes_attribute() -> None:
    assert count_detections(object()) == 0


def test_load_model_without_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Файл весов не найден"):
        load_model(tmp_path / "best.pt")


def test_load_model_with_lfs_pointer(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_text(LFS_POINTER, encoding="utf-8")
    assert is_lfs_pointer(weights)
    with pytest.raises(FileNotFoundError, match="git lfs pull"):
        load_model(weights)


def test_predict_image_passes_settings(
    tmp_path: Path,
    make_image: Callable[..., Path],
    make_model: Callable[..., Any],
) -> None:
    image = make_image(tmp_path / "photo.png")
    model = make_model()
    params = PredictConfig(conf=0.4, iou=0.5, imgsz=320, device="cpu")

    result = predict_image(model, image, params)

    assert result is model.result
    source, kwargs = model.calls[0]
    assert source == str(image)
    assert kwargs["conf"] == pytest.approx(0.4)
    assert kwargs["iou"] == pytest.approx(0.5)
    assert kwargs["imgsz"] == 320
    assert kwargs["device"] == "cpu"
    assert kwargs["verbose"] is False


def test_predict_image_auto_device(
    tmp_path: Path,
    make_image: Callable[..., Path],
    make_model: Callable[..., Any],
) -> None:
    model = make_model()
    predict_image(model, make_image(tmp_path / "a.png"), PredictConfig())
    assert model.calls[0][1]["device"] is None


def test_predict_image_missing_file(
    tmp_path: Path, make_model: Callable[..., Any]
) -> None:
    model = make_model()
    with pytest.raises(FileNotFoundError, match="не найдено"):
        predict_image(model, tmp_path / "missing.png", PredictConfig())
    assert model.calls == []


def test_detect_counts_and_converts_to_rgb(
    tmp_path: Path,
    make_image: Callable[..., Path],
    make_model: Callable[..., Any],
) -> None:
    image = make_image(tmp_path / "photo.png")
    prediction = detect(make_model(count=7), image, PredictConfig())

    assert prediction.image_path == image
    assert prediction.count == 7
    # фальшивая модель рисует синий цвет в BGR, в RGB он в третьем канале
    assert prediction.annotated[0, 0].tolist() == [0, 0, 255]
