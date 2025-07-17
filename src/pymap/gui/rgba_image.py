"""Wrapper for a pixmap and pixmap items that are a single RGBA image."""

from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
)

from pymap.gui.render import ndarray_to_QImage
from pymap.gui.types import RGBAImage


class QRGBAImage:
    """A wrapper for a pixmap and pixmap items that are a single RGBA image."""

    def __init__(self, rgba_image: RGBAImage):
        """Initialize with an RGBA image."""
        self.rgba_image = rgba_image
        self.pixmap = QPixmap.fromImage(ndarray_to_QImage(rgba_image))
        self.item: QGraphicsPixmapItem = QGraphicsPixmapItem(self.pixmap)

    def set_rectangle(self, image: RGBAImage, x: int, y: int) -> None:
        """Paints the given image at the specified position on the pixmap."""
        height, width = image.shape[:2]
        assert self.pixmap is not None, 'Pixmap is not initialized'
        assert self.rgba_image is not None, 'RGBA image is not initialized'
        self.rgba_image[y : y + height, x : x + width] = image
        with QPainter(self.pixmap) as painter:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawImage(x, y, ndarray_to_QImage(image))
        self.item.setPixmap(self.pixmap)
