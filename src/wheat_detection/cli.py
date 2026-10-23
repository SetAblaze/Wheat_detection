"""Командная строка проекта.

Примеры::

    wheat-detection predict data/arvalis_3/photo.png
    wheat-detection predict --random --show
    wheat-detection compare --random
    wheat-detection split-data
    wheat-detection train --epochs 50
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from wheat_detection import __version__
from wheat_detection.config import Settings, load_settings, with_overrides
from wheat_detection.data import pick_random_image, prepare_split
from wheat_detection.inference import Prediction, detect, load_model
from wheat_detection.train import train
from wheat_detection.visualization import (
    build_comparison,
    load_rgb,
    save_figure,
    save_image,
    show_file,
)

logger = logging.getLogger("wheat_detection")

Handler = Callable[[argparse.Namespace, Settings], int]

PROMPT = "Введите путь к фото (или перетащите файл в окно): "


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа: разобрать аргументы и выполнить команду.

    Возвращает код завершения: 0 при успехе, 1 при ошибке в данных
    или настройках, 130 при остановке через Ctrl+C.
    """
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)

    handler: Handler = args.handler
    try:
        settings = load_settings(args.config)
        return handler(args, settings)
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        logger.debug("Подробности ошибки:", exc_info=True)
        return 1
    except KeyboardInterrupt:
        logger.warning("Остановлено пользователем")
        return 130


def build_parser() -> argparse.ArgumentParser:
    """Описать команды и их аргументы."""
    parser = argparse.ArgumentParser(
        prog="wheat-detection",
        description="Поиск колосьев пшеницы на фотографиях поля (YOLOv8).",
        parents=[_common_options(suppress_defaults=False)],
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="КОМАНДА"
    )
    # Общие флаги -c и -v работают и до команды, и после неё
    common = _common_options(suppress_defaults=True)

    predict = commands.add_parser(
        "predict", help="найти колосья на фото", parents=[common]
    )
    _add_source_arguments(predict)
    _add_predict_arguments(predict)
    predict.set_defaults(handler=run_predict)

    compare = commands.add_parser(
        "compare", help="сохранить картинку «до и после»", parents=[common]
    )
    _add_source_arguments(compare)
    _add_predict_arguments(compare)
    compare.set_defaults(handler=run_compare)

    split = commands.add_parser(
        "split-data",
        help="разбить фото для обучения на train и val",
        parents=[common],
    )
    split.add_argument("--val-fraction", type=float, help="доля фото в val")
    split.add_argument("--seed", type=int, help="seed для разбиения")
    split.set_defaults(handler=run_split)

    train_parser = commands.add_parser(
        "train", help="обучить модель", parents=[common]
    )
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch", type=int)
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--device", help="auto, cpu или номер GPU")
    train_parser.add_argument(
        "--base-model", help="исходные веса, например yolov8n.pt"
    )
    train_parser.add_argument("--name", help="имя запуска в runs/train")
    train_parser.set_defaults(handler=run_train)
    return parser


def setup_logging(verbose: bool = False) -> None:
    """Настроить вывод сообщений пакета в консоль."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"
        )
    )
    logger.handlers[:] = [handler]
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False


def run_predict(args: argparse.Namespace, settings: Settings) -> int:
    """Команда predict: сохранить фото с найденными колосьями."""
    settings = _apply_predict_overrides(args, settings)
    prediction = _detect(args, settings)
    output_dir = args.output_dir or settings.paths.runs_dir / "predict"
    output = save_image(
        prediction.annotated,
        output_dir / f"{prediction.image_path.stem}_pred.jpg",
    )
    _report(prediction, output, show=args.show)
    return 0


def run_compare(args: argparse.Namespace, settings: Settings) -> int:
    """Команда compare: исходное фото и фото с рамками рядом."""
    settings = _apply_predict_overrides(args, settings)
    prediction = _detect(args, settings)
    figure = build_comparison(
        original=load_rgb(prediction.image_path),
        annotated=prediction.annotated,
        image_name=prediction.image_path.name,
        count=prediction.count,
    )
    output_dir = args.output_dir or settings.paths.runs_dir / "compare"
    output = save_figure(
        figure, output_dir / f"{prediction.image_path.stem}_compare.png"
    )
    _report(prediction, output, show=args.show)
    return 0


def run_split(args: argparse.Namespace, settings: Settings) -> int:
    """Команда split-data: разбить фото на train и val."""
    settings = with_overrides(
        settings, "train", val_fraction=args.val_fraction, seed=args.seed
    )
    prepare_split(
        images_dir=settings.paths.train_images_dir,
        splits_dir=settings.paths.splits_dir,
        dataset_template=settings.paths.dataset_template,
        val_fraction=settings.train.val_fraction,
        seed=settings.train.seed,
    )
    return 0


def run_train(args: argparse.Namespace, settings: Settings) -> int:
    """Команда train: обучить модель."""
    settings = with_overrides(
        settings,
        "train",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        base_model=args.base_model,
    )
    train(settings, run_name=args.name)
    return 0


def ask_image_path(read: Callable[[str], str] | None = None) -> Path:
    """Спросить путь к фото, как в исходном скрипте detect_drag.py.

    При перетаскивании файла в окно терминала путь часто приходит
    в кавычках, они отбрасываются. ``read`` нужен для тестов,
    по умолчанию используется ``input``.
    """
    reader = input if read is None else read
    try:
        raw = reader(PROMPT)
    except EOFError:
        raw = ""
    cleaned = raw.strip().strip("\"'").strip()
    if not cleaned:
        raise ValueError(
            "Не указан путь к фото. Передайте его аргументом "
            "или используйте --random"
        )
    return Path(cleaned)


def _common_options(suppress_defaults: bool) -> argparse.ArgumentParser:
    """Флаги, общие для всех команд.

    У подкоманд значения по умолчанию подавлены (SUPPRESS), иначе они
    затёрли бы флаг, указанный перед командой.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-c",
        "--config",
        type=Path,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="YAML с настройками (по умолчанию configs/config.yaml)",
    )
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS if suppress_defaults else False,
        help="подробный вывод",
    )
    return common


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "image",
        nargs="?",
        type=Path,
        help="путь к фото; если не указан, программа спросит его",
    )
    source.add_argument(
        "--random",
        action="store_true",
        help="взять случайное фото из папки с примерами",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="папка для --random (по умолчанию paths.sample_images_dir)",
    )
    parser.add_argument("--seed", type=int, help="seed для --random")


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", type=Path, help="файл весов модели")
    parser.add_argument("--conf", type=float, help="порог уверенности")
    parser.add_argument("--device", help="auto, cpu или номер GPU")
    parser.add_argument("--output-dir", type=Path, help="папка для результата")
    parser.add_argument(
        "--show", action="store_true", help="открыть результат на экране"
    )


def _apply_predict_overrides(
    args: argparse.Namespace, settings: Settings
) -> Settings:
    settings = with_overrides(settings, "paths", weights=args.weights)
    return with_overrides(
        settings, "predict", conf=args.conf, device=args.device
    )


def _resolve_source(args: argparse.Namespace, settings: Settings) -> Path:
    if args.random:
        images_dir = args.images_dir or settings.paths.sample_images_dir
        image = pick_random_image(images_dir, seed=args.seed)
        logger.info("Случайное фото: %s", image)
        return image
    if args.image is not None:
        return Path(args.image)
    return ask_image_path()


def _detect(args: argparse.Namespace, settings: Settings) -> Prediction:
    image_path = _resolve_source(args, settings)
    if not image_path.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {image_path}")
    logger.info("Обработка: %s", image_path.name)
    model = load_model(settings.paths.weights)
    return detect(model, image_path, settings.predict)


def _report(prediction: Prediction, output: Path, show: bool) -> None:
    logger.info("Найдено колосьев: %d", prediction.count)
    logger.info("Результат сохранён: %s", output)
    if show:
        show_file(output)
