import logging
from pathlib import Path

import pytest
import yaml

from wheat_detection.config import ConfigError
from wheat_detection.data import (
    find_images_without_labels,
    label_path_for,
    list_images,
    pick_random_image,
    prepare_split,
    split_images,
)


@pytest.fixture
def template(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text("names:\n  0: wheat_head\n", encoding="utf-8")
    return path


def test_list_images_filters_and_sorts(images_dir: Path) -> None:
    names = [path.name for path in list_images(images_dir)]
    assert names == ["a.jpg", "b.png", "c.JPEG"]


def test_list_images_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="не найдена"):
        list_images(tmp_path / "missing")


def test_pick_random_image_is_reproducible(images_dir: Path) -> None:
    first = pick_random_image(images_dir, seed=7)
    assert first == pick_random_image(images_dir, seed=7)
    assert first in list_images(images_dir)


def test_pick_random_image_from_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="нет изображений"):
        pick_random_image(tmp_path / "empty")


@pytest.mark.parametrize(
    ("image", "label"),
    [
        ("data/train/images/a.png", "data/train/labels/a.txt"),
        ("images/x/images/b.jpg", "images/x/labels/b.txt"),
        ("data/arvalis_3/c.png", "data/arvalis_3/c.txt"),
    ],
)
def test_label_path_for(image: str, label: str) -> None:
    assert label_path_for(Path(image)) == Path(label)


def test_find_images_without_labels(yolo_dataset: Path) -> None:
    images = list_images(yolo_dataset / "train" / "images")
    missing = find_images_without_labels(images)
    assert [path.name for path in missing] == ["img_09.png"]


def test_split_is_disjoint_complete_and_reproducible() -> None:
    images = [Path(f"{index}.png") for index in range(10)]
    split = split_images(images, val_fraction=0.2, seed=42)

    assert len(split.val) == 2
    assert len(split.train) == 8
    assert not set(split.train) & set(split.val)
    assert set(split.train) | set(split.val) == set(images)
    assert split_images(images[::-1], val_fraction=0.2, seed=42) == split


def test_split_depends_on_seed() -> None:
    images = [Path(f"{index}.png") for index in range(20)]
    splits = {split_images(images, 0.5, seed=seed).val for seed in range(5)}
    assert len(splits) > 1


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_split_keeps_at_least_one_image_on_each_side(fraction: float) -> None:
    split = split_images([Path("a.png"), Path("b.png")], fraction, seed=0)
    assert len(split.train) == 1
    assert len(split.val) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_bad_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="между 0 и 1"):
        split_images([Path("a.png"), Path("b.png")], fraction, seed=0)


def test_split_needs_two_images() -> None:
    with pytest.raises(ValueError, match="два изображения"):
        split_images([Path("a.png")], 0.5, seed=0)


def test_prepare_split_writes_files_for_yolo(
    yolo_dataset: Path, template: Path
) -> None:
    files = prepare_split(
        images_dir=yolo_dataset / "train" / "images",
        splits_dir=yolo_dataset / "splits",
        dataset_template=template,
        val_fraction=0.2,
        seed=42,
    )

    train_lines = files.train_list.read_text(encoding="utf-8").splitlines()
    val_lines = files.val_list.read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 8
    assert len(val_lines) == 2
    assert not set(train_lines) & set(val_lines)
    for line in train_lines + val_lines:
        assert Path(line).is_absolute()
        assert Path(line).is_file()

    dataset = yaml.safe_load(files.dataset_yaml.read_text(encoding="utf-8"))
    assert dataset["names"] == {0: "wheat_head"}
    assert Path(dataset["train"]) == files.train_list.resolve()
    assert Path(dataset["val"]) == files.val_list.resolve()


def test_prepare_split_warns_about_missing_labels(
    yolo_dataset: Path, template: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="wheat_detection"):
        prepare_split(
            yolo_dataset / "train" / "images",
            yolo_dataset / "splits",
            template,
            val_fraction=0.2,
            seed=1,
        )
    assert "нет файла разметки" in caplog.text


def test_prepare_split_on_empty_directory(
    tmp_path: Path, template: Path
) -> None:
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="нет фото"):
        prepare_split(tmp_path / "images", tmp_path / "s", template, 0.2, 0)


def test_template_without_classes_is_rejected(
    yolo_dataset: Path, tmp_path: Path
) -> None:
    template = tmp_path / "bad.yaml"
    template.write_text("nc: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="names"):
        prepare_split(
            yolo_dataset / "train" / "images",
            tmp_path / "splits",
            template,
            0.2,
            0,
        )
