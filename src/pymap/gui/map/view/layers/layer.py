"""Layers on the map view."""

from __future__ import annotations

from abc import abstractmethod
from enum import IntFlag, auto, unique
from typing import TYPE_CHECKING

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
)

from pymap.gui.rgba_image import QRGBAImage
from pymap.gui.types import RGBAImage

if TYPE_CHECKING:
    from ..map_view import MapView


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
        self.qrgba_image: QRGBAImage | None = None

    @property
    def pixmap(self) -> QPixmap | None:
        """Get the pixmap of the RGBA image."""
        if self.qrgba_image is not None:
            return self.qrgba_image.pixmap
        return None

    def set_rgba_image(self, rgba_image: RGBAImage) -> None:
        """Set the RGBA image for the layer."""
        self.qrgba_image = QRGBAImage(rgba_image)
        self.item = self.qrgba_image.item

    def update_rectangle_with_image(self, image: RGBAImage, x: int, y: int):
        """Updates the rectangle with the given image at the specified position."""
        assert self.qrgba_image is not None, 'RGBA image is not initialized'
        self.qrgba_image.set_rectangle(image, x, y)


@unique
class VisibleLayer(IntFlag):
    """Layers that can be visible."""

    BLOCKS = auto()
    LEVELS = auto()
    BORDER_EFFECT = auto()
    SMART_SHAPE = auto()
    EVENTS = auto()
    SELECTED_EVENT = auto()
    GRID = auto()
    CONNECTION_RECTANGLES = auto()
