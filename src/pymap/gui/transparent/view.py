"""Module for a QGraphicsView that can display a transparent background."""

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsView

from pymap.gui import render

from .transparent_tile import get_transparent_background


class QGraphicsViewWithTransparentBackground(QGraphicsView):
    """A QGraphicsView that can display a transparent background."""

    def get_transparent_background(
        self,
        width: int,
        height: int,
        scaled_width: int | None = None,
        scaled_height: int | None = None,
    ) -> QGraphicsPixmapItem:
        """Get the transparent background item.

        Args:
            width (int): The width of the background.
            height (int): The height of the background.
            scaled_width (int | None, optional): The scaled width of the background.
                Defaults to None.
            scaled_height (int | None, optional): The scaled height of the background.

        Returns:
            QGraphicsPixmapItem: The item containing the transparent background.
        """
        transparent_background_pixmap = QPixmap.fromImage(
            render.ndarray_to_QImage(get_transparent_background(width, height))
        )
        if scaled_width is not None and scaled_height is not None:
            transparent_background_pixmap = transparent_background_pixmap.scaled(
                scaled_width, scaled_height
            )
        else:
            scaled_width, scaled_height = (
                transparent_background_pixmap.width(),
                transparent_background_pixmap.height(),
            )
        item = QGraphicsPixmapItem(transparent_background_pixmap)
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        return item
