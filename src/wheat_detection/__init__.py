"""Детекция колосьев пшеницы на фотографиях поля (YOLOv8)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wheat-detection")
except PackageNotFoundError:  # код запущен без установки пакета
    __version__ = "0.0.0"

__all__ = ["__version__"]
