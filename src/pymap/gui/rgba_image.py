"""Wrapper for a pixmap and pixmap items that are a single RGBA image."""

from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
)

from pymap.gui.render import ndarray_to_QImage
from pymap.gui.types import RGBAImage


class QRGBAImage:
    """A wrapper for a pixmap and pixmap items that are a single RGBA image."""

    def __init__(
        self,
        rgba_image: RGBAImage,
        scaled_width: int | None = None,
        scaled_height: int | None = None,
        scaling_factor: float | None = None,
    ):
        """Initialize with an RGBA image."""
        self.rgba_image = rgba_image
        if scaling_factor is not None:
            assert scaled_width is None and scaled_height is None, (
                'Cannot set both scaling factor and scaled dimensions'
            )
            scaled_width = int(rgba_image.shape[1] * scaling_factor)
            scaled_height = int(rgba_image.shape[0] * scaling_factor)
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height
        self.pixmap = QPixmap.fromImage(ndarray_to_QImage(rgba_image))
        if self.scaled_width is not None and self.scaled_height is not None:
            self.item = QGraphicsPixmapItem(
                self.pixmap.scaled(self.scaled_width, self.scaled_height)
            )
        else:
            self.item = QGraphicsPixmapItem(self.pixmap)

    @property
    def width(self) -> int:
        """Return the width of the RGBA image."""
        return self.rgba_image.shape[1]

    @property
    def height(self) -> int:
        """Return the height of the RGBA image."""
        return self.rgba_image.shape[0]

    def set_rectangle(self, image: RGBAImage, x: int, y: int) -> None:
        """Paints the given image at the specified position on the pixmap."""
        height, width = image.shape[:2]
        assert self.pixmap is not None, 'Pixmap is not initialized'
        assert self.rgba_image is not None, 'RGBA image is not initialized'
        self.rgba_image[y : y + height, x : x + width] = image
        with QPainter(self.pixmap) as painter:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            painter.drawImage(x, y, ndarray_to_QImage(image))
        if self.scaled_width is not None and self.scaled_height is not None:
            self.item.setPixmap(
                self.pixmap.scaled(self.scaled_width, self.scaled_height)
            )
        else:
            self.item.setPixmap(self.pixmap)
