from pathlib import Path

import pytest

from wheat_detection.config import (
    ConfigError,
    PredictConfig,
    Settings,
    TrainConfig,
    load_settings,
    with_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_config(directory: Path, text: str) -> Path:
    path = directory / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_when_config_file_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_settings(environ={}) == Settings()


def test_repository_config_is_valid() -> None:
    settings = load_settings(REPO_ROOT / "configs" / "config.yaml", environ={})
    assert settings.paths.weights == Path("models/best.pt")
    assert settings.train.epochs == 50
    assert settings.train.val_fraction == pytest.approx(0.2)


def test_yaml_overrides_defaults(tmp_path: Path) -> None:
    config = write_config(
        tmp_path, "predict:\n  conf: 0.5\npaths:\n  weights: other.pt\n"
    )
    settings = load_settings(config, environ={})
    assert settings.predict.conf == pytest.approx(0.5)
    assert settings.paths.weights == Path("other.pt")
    assert settings.train == TrainConfig()


def test_environment_overrides_yaml(tmp_path: Path) -> None:
    config = write_config(tmp_path, "predict:\n  conf: 0.5\n")
    environ = {
        "WHEAT_PREDICT__CONF": "0.6",
        "WHEAT_TRAIN__EPOCHS": "3",
        "WHEAT_TRAIN__DEVICE": "cpu",
    }
    settings = load_settings(config, environ=environ)
    assert settings.predict.conf == pytest.approx(0.6)
    assert settings.train.epochs == 3
    assert settings.train.device == "cpu"


def test_unrelated_environment_variables_are_ignored(tmp_path: Path) -> None:
    config = write_config(tmp_path, "")
    environ = {"WHEAT_TOKEN": "x", "PATH": "/usr/bin"}
    assert load_settings(config, environ=environ) == Settings()


def test_numeric_device_becomes_string(tmp_path: Path) -> None:
    config = write_config(tmp_path, "train:\n  device: 0\n")
    assert load_settings(config, environ={}).train.device == "0"


def test_explicit_missing_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_settings(tmp_path / "missing.yaml", environ={})


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("unknown:\n  a: 1\n", "неизвестный раздел"),
        ("predict:\n  confidence: 0.5\n", "Неизвестные параметры"),
        ("train:\n  epochs: many\n", "train.epochs: некорректное"),
        ("train:\n  epochs: 2.5\n", "train.epochs: некорректное"),
        ("train:\n  device: false\n", "train.device: некорректное"),
        ("train:\n  epochs: 0\n", "train.epochs должно"),
        ("train:\n  val_fraction: 1.0\n", "train.val_fraction"),
        ("predict:\n  conf: 0\n", "predict.conf"),
        ("predict: 5\n", "не словарь"),
        ("- just\n- a list\n", "ожидался словарь"),
    ],
)
def test_invalid_config_is_rejected(
    tmp_path: Path, text: str, message: str
) -> None:
    config = write_config(tmp_path, text)
    with pytest.raises(ConfigError, match=message):
        load_settings(config, environ={})


def test_unknown_environment_key_is_rejected(tmp_path: Path) -> None:
    config = write_config(tmp_path, "")
    with pytest.raises(ConfigError, match="confidence"):
        load_settings(config, environ={"WHEAT_PREDICT__CONFIDENCE": "0.3"})


def test_all_errors_are_reported_together(tmp_path: Path) -> None:
    config = write_config(tmp_path, "train:\n  epochs: 0\n  imgsz: -1\n")
    with pytest.raises(ConfigError) as error:
        load_settings(config, environ={})
    assert "train.epochs" in str(error.value)
    assert "train.imgsz" in str(error.value)


def test_with_overrides_skips_none_values() -> None:
    settings = Settings()
    updated = with_overrides(settings, "predict", conf=0.4, device=None)
    assert updated.predict == PredictConfig(conf=0.4)
    assert with_overrides(settings, "predict", conf=None) is settings


def test_with_overrides_validates_result() -> None:
    with pytest.raises(ConfigError, match="predict.conf"):
        with_overrides(Settings(), "predict", conf=2.0)


def test_with_overrides_rejects_unknown_section() -> None:
    with pytest.raises(ConfigError, match="Неизвестный раздел"):
        with_overrides(Settings(), "model", conf=0.5)
