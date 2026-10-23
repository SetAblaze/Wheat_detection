import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from wheat_detection import cli
from wheat_detection.cli import ask_image_path, main


@pytest.fixture
def fake_model(
    monkeypatch: pytest.MonkeyPatch, make_model: Callable[..., Any]
) -> Any:
    """Подменить загрузку YOLO фальшивой моделью, которая находит 4 колоса."""
    model = make_model(count=4)
    monkeypatch.setattr(cli, "load_model", lambda weights: model)
    return model


def test_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for command in ("predict", "compare", "split-data", "train"):
        assert command in output


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.startswith("wheat-detection ")


def test_command_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2


@pytest.mark.parametrize("verbose_first", [True, False])
def test_verbose_flag_before_or_after_command(
    project_dir: Path,
    fake_model: Any,
    make_image: Callable[..., Path],
    verbose_first: bool,
) -> None:
    image = str(make_image(project_dir / "photo.png"))
    args = ["-v", "predict", image] if verbose_first else ["predict", image]
    if not verbose_first:
        args.append("-v")

    assert main(args) == 0
    assert logging.getLogger("wheat_detection").level == logging.DEBUG


def test_config_flag_after_command(
    project_dir: Path, fake_model: Any, make_image: Callable[..., Path]
) -> None:
    config = project_dir / "custom.yaml"
    config.write_text("predict:\n  conf: 0.45\n", encoding="utf-8")
    image = make_image(project_dir / "photo.png")

    assert main(["predict", str(image), "-c", str(config)]) == 0
    assert fake_model.calls[0][1]["conf"] == pytest.approx(0.45)


def test_image_and_random_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["predict", "photo.png", "--random"])
    assert exit_info.value.code == 2


def test_predict_saves_annotated_image(
    project_dir: Path,
    fake_model: Any,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = make_image(project_dir / "photo.png")

    assert main(["predict", str(image)]) == 0

    assert (project_dir / "runs" / "predict" / "photo_pred.jpg").is_file()
    assert "Найдено колосьев: 4" in capsys.readouterr().err


def test_predict_random_image_from_sample_dir(
    project_dir: Path, fake_model: Any, make_image: Callable[..., Path]
) -> None:
    for name in ("one.png", "two.png"):
        make_image(project_dir / "data" / "arvalis_3" / name)

    assert main(["predict", "--random", "--seed", "0"]) == 0

    outputs = list((project_dir / "runs" / "predict").glob("*_pred.jpg"))
    assert len(outputs) == 1


def test_predict_asks_for_path_when_not_given(
    project_dir: Path,
    fake_model: Any,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = make_image(project_dir / "dragged photo.png")
    monkeypatch.setattr("builtins.input", lambda prompt: f' "{image}" ')

    assert main(["predict"]) == 0

    expected = project_dir / "runs" / "predict" / "dragged photo_pred.jpg"
    assert expected.is_file()


def test_ask_image_path_without_input() -> None:
    def closed_stdin(prompt: str) -> str:
        raise EOFError

    with pytest.raises(ValueError, match="--random"):
        ask_image_path(closed_stdin)


def test_cli_arguments_reach_the_model(
    project_dir: Path, fake_model: Any, make_image: Callable[..., Path]
) -> None:
    image = make_image(project_dir / "photo.png")

    args = ["predict", str(image), "--conf", "0.5", "--device", "cpu"]
    assert main(args) == 0

    kwargs = fake_model.calls[0][1]
    assert kwargs["conf"] == pytest.approx(0.5)
    assert kwargs["device"] == "cpu"


def test_env_file_is_loaded(
    project_dir: Path, fake_model: Any, make_image: Callable[..., Path]
) -> None:
    (project_dir / ".env").write_text(
        "WHEAT_PREDICT__CONF=0.33\n", encoding="utf-8"
    )
    image = make_image(project_dir / "photo.png")

    assert main(["predict", str(image)]) == 0

    assert fake_model.calls[0][1]["conf"] == pytest.approx(0.33)


def test_missing_image_is_reported(
    project_dir: Path,
    fake_model: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["predict", "missing.png"]) == 1
    assert "Изображение не найдено" in capsys.readouterr().err
    assert fake_model.calls == []


def test_missing_weights_are_reported(
    project_dir: Path,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = make_image(project_dir / "photo.png")
    assert main(["predict", str(image)]) == 1
    assert "Файл весов не найден" in capsys.readouterr().err


def test_bad_value_is_reported(
    project_dir: Path,
    fake_model: Any,
    make_image: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = make_image(project_dir / "photo.png")
    assert main(["predict", str(image), "--conf", "1.5"]) == 1
    assert "predict.conf" in capsys.readouterr().err


def test_compare_saves_figure(
    project_dir: Path, fake_model: Any, make_image: Callable[..., Path]
) -> None:
    image = make_image(project_dir / "photo.png")

    assert main(["compare", str(image), "--output-dir", "out"]) == 0

    assert (project_dir / "out" / "photo_compare.png").is_file()


def test_split_data_command(project_dir: Path, yolo_dataset: Path) -> None:
    args = ["split-data", "--val-fraction", "0.3", "--seed", "1"]
    assert main(args) == 0

    splits = yolo_dataset / "splits"
    train_list = (splits / "train.txt").read_text(encoding="utf-8")
    val_list = (splits / "val.txt").read_text(encoding="utf-8")
    assert len(train_list.splitlines()) == 7
    assert len(val_list.splitlines()) == 3
    assert (splits / "dataset.yaml").is_file()


def test_train_command(
    project_dir: Path,
    yolo_dataset: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            calls["weights"] = weights
            self.trainer = SimpleNamespace(best="runs/train/smoke/best.pt")

        def train(self, **kwargs: Any) -> None:
            calls["train"] = kwargs

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    args = ["train", "--epochs", "1", "--device", "cpu", "--name", "smoke"]
    assert main(args) == 0

    assert calls["weights"] == "yolov8n.pt"
    kwargs = calls["train"]
    assert kwargs["epochs"] == 1
    assert kwargs["device"] == "cpu"
    assert kwargs["name"] == "smoke"
    assert kwargs["seed"] == 42

    dataset = yaml.safe_load(Path(kwargs["data"]).read_text(encoding="utf-8"))
    assert dataset["names"] == {0: "wheat_head"}
    val_images = Path(dataset["val"]).read_text(encoding="utf-8").splitlines()
    assert len(val_images) == 2
