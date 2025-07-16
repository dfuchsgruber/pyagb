"""Layers on the map view."""

from __future__ import annotations

from abc import abstractmethod
from enum import IntFlag, auto, unique
from typing import TYPE_CHECKING

from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
)

from pymap.gui.render import ndarray_to_QImage
from pymap.gui.types import RGBAImage

if TYPE_CHECKING:
    from .map_view import MapView


from abc import ABC


class MapViewLayer(ABC):
    """A layer in the map view."""

    def __init__(self, view: MapView):
        """Initialize the layer with an RGBA image."""
        self.view = view
        self.item: QGraphicsItem | None = None

    @abstractmethod
    def load_map(self) -> None:
        """Load the map."""
        raise NotImplementedError('Subclasses must implement this method.')


class MapViewLayerRGBAImage(MapViewLayer):
    """A layer in the map view that is a single RGBA image."""

    def __init__(self, view: MapView):
        """Initialize the layer with an RGBA image."""
        super().__init__(view)
        self.rgba_image: RGBAImage | None = None
        self.pixmap: QPixmap | None = None

    def set_rgba_image(self, rgba_image: RGBAImage) -> None:
        """Set the RGBA image for the layer."""
        self.rgba_image = rgba_image
        self.pixmap = QPixmap.fromImage(ndarray_to_QImage(rgba_image))
        self.item = QGraphicsPixmapItem(self.pixmap)

    def update_rectangle_with_image(self, image: RGBAImage, x: int, y: int):
        """Updates the rectangle with the given image at the specified position."""
        height, width = image.shape[:2]
        assert self.pixmap is not None, 'Pixmap is not initialized'
        assert self.rgba_image is not None, 'RGBA image is not initialized'
        self.rgba_image[y : y + height, x : x + width] = image
        with QPainter(self.pixmap) as painter:
            # Clear the area before drawing the new image
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            # Draw the image
            painter.drawImage(
                x,
                y,
                ndarray_to_QImage(image),
            )
        assert isinstance(self.item, QGraphicsPixmapItem), (
            'Item is not a QGraphicsPixmapItem'
        )
        self.item.setPixmap(self.pixmap)


@unique
class VisibleLayer(IntFlag):
    """Layers that can be visible."""

    BLOCKS = auto()
    LEVELS = auto()
    SMART_SHAPE = auto()
    EVENTS = auto()
    SELECTED_EVENT = auto()
    GRID = auto()
    CONNECTION_RECTANGLES = auto()
