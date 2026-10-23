"""Настройки проекта.

Источники настроек, каждый следующий перекрывает предыдущий:

1. значения по умолчанию из этого модуля;
2. YAML-файл (по умолчанию ``configs/config.yaml``);
3. переменные окружения вида ``WHEAT_<РАЗДЕЛ>__<ПАРАМЕТР>``,
   например ``WHEAT_PREDICT__CONF=0.4`` (в том числе из файла ``.env``);
4. аргументы командной строки.

Относительные пути считаются от текущей папки, то есть от корня проекта,
из которого запускаются команды.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_type_hints

import yaml

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

DEFAULT_CONFIG_PATH = Path("configs") / "config.yaml"
ENV_PREFIX = "WHEAT_"
ENV_SEPARATOR = "__"

_T = TypeVar("_T", bound="DataclassInstance")


class ConfigError(ValueError):
    """Ошибка в настройках: неизвестный параметр или плохое значение."""


@dataclass(frozen=True)
class PathsConfig:
    """Пути к данным, весам модели и результатам запусков."""

    train_images_dir: Path = Path("data/train/images")
    sample_images_dir: Path = Path("data/arvalis_3")
    splits_dir: Path = Path("data/splits")
    dataset_template: Path = Path("configs/dataset.yaml")
    weights: Path = Path("models/best.pt")
    runs_dir: Path = Path("runs")


@dataclass(frozen=True)
class TrainConfig:
    """Параметры обучения."""

    base_model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    workers: int = 8
    seed: int = 42
    val_fraction: float = 0.2


@dataclass(frozen=True)
class PredictConfig:
    """Параметры поиска колосьев на фото."""

    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    device: str = "auto"
    max_det: int = 300


@dataclass(frozen=True)
class Settings:
    """Все настройки проекта."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)


_SECTIONS: dict[str, type[DataclassInstance]] = {
    "paths": PathsConfig,
    "train": TrainConfig,
    "predict": PredictConfig,
}


def load_settings(
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Собрать настройки из значений по умолчанию, YAML и окружения.

    Если ``config_path`` не передан, читается ``configs/config.yaml``,
    а при его отсутствии используются значения по умолчанию. Явно
    указанный, но отсутствующий файл считается ошибкой.
    """
    raw: dict[str, dict[str, Any]] = {name: {} for name in _SECTIONS}

    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    if path.is_file():
        _merge(raw, _read_yaml(path), source=str(path))
    elif config_path is not None:
        raise FileNotFoundError(f"Файл настроек не найден: {config_path}")

    env = os.environ if environ is None else environ
    _merge(raw, _read_env(env), source="переменные окружения")

    settings = Settings(
        paths=_build_section(PathsConfig, "paths", raw["paths"]),
        train=_build_section(TrainConfig, "train", raw["train"]),
        predict=_build_section(PredictConfig, "predict", raw["predict"]),
    )
    validate(settings)
    return settings


def with_overrides(
    settings: Settings, section: str, **values: Any
) -> Settings:
    """Вернуть копию настроек с новыми значениями в разделе ``section``.

    Значения ``None`` пропускаются: так аргументы командной строки,
    которые пользователь не указал, не затирают настройки из файла.
    """
    updates = {
        key: value for key, value in values.items() if value is not None
    }
    if not updates:
        return settings
    if section not in _SECTIONS:
        raise ConfigError(f"Неизвестный раздел настроек: {section}")
    current = getattr(settings, section)
    updated = replace(settings, **{section: replace(current, **updates)})
    validate(updated)
    return updated


def validate(settings: Settings) -> None:
    """Проверить значения и выдать все найденные ошибки разом."""
    train, predict = settings.train, settings.predict
    checks = [
        (train.epochs > 0, "train.epochs должно быть больше 0"),
        (train.imgsz > 0, "train.imgsz должно быть больше 0"),
        (
            train.batch > 0 or train.batch == -1,
            "train.batch: положительное число или -1 (автоподбор)",
        ),
        (train.workers >= 0, "train.workers не может быть меньше 0"),
        (
            0 < train.val_fraction < 1,
            "train.val_fraction должно быть строго между 0 и 1",
        ),
        (0 < predict.conf <= 1, "predict.conf должно быть в пределах (0, 1]"),
        (0 < predict.iou <= 1, "predict.iou должно быть в пределах (0, 1]"),
        (predict.imgsz > 0, "predict.imgsz должно быть больше 0"),
        (predict.max_det > 0, "predict.max_det должно быть больше 0"),
    ]
    errors = [message for passed, message in checks if not passed]
    if errors:
        raise ConfigError("Ошибки в настройках:\n  - " + "\n  - ".join(errors))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        sections = ", ".join(_SECTIONS)
        raise ConfigError(f"{path}: ожидался словарь с разделами {sections}")
    return data


def _read_env(environ: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Выбрать переменные ``WHEAT_<РАЗДЕЛ>__<ПАРАМЕТР>``."""
    result: dict[str, dict[str, str]] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        section, separator, name = (
            key[len(ENV_PREFIX) :].lower().partition(ENV_SEPARATOR)
        )
        if separator and section and name:
            result.setdefault(section, {})[name] = value
    return result


def _merge(
    raw: dict[str, dict[str, Any]],
    data: Mapping[str, Any],
    source: str,
) -> None:
    for section, values in data.items():
        if section not in raw:
            raise ConfigError(f"{source}: неизвестный раздел '{section}'")
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise ConfigError(f"{source}: раздел '{section}' не словарь")
        raw[section].update(values)


def _build_section(
    cls: type[_T],
    section: str,
    values: Mapping[str, Any],
) -> _T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(
            f"Неизвестные параметры в разделе '{section}': {names}"
        )
    hints = get_type_hints(cls)
    kwargs = {
        name: _coerce(f"{section}.{name}", value, hints[name])
        for name, value in values.items()
    }
    return cls(**kwargs)


def _coerce(name: str, value: Any, target: Any) -> Any:
    """Привести значение из YAML или окружения к типу поля."""
    try:
        if isinstance(value, bool):
            raise TypeError("логическое значение вместо числа или строки")
        if target is Path:
            return Path(value)
        if target is int:
            number = float(value)
            if not number.is_integer():
                raise ValueError("ожидалось целое число")
            return int(number)
        if target is float:
            return float(value)
        if target is str:
            return str(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"{name}: некорректное значение {value!r}"
        ) from error
    raise ConfigError(f"{name}: неподдерживаемый тип {target!r}")
