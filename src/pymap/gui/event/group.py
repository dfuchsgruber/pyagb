"""QGraphicsItemGroup subclasses for events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui.render import ndarray_to_QImage

if TYPE_CHECKING:
    from .map_scene import MapScene


class EventGroupRectangular(QGraphicsItemGroup):
    """Subclass for an rectangular event group."""

    def __init__(self, map_scene: MapScene, event_type: PymapEventConfigType):
        """Initializes the group.

        Args:
            map_scene (MapScene): The map scene.
            event_type (PymapEventConfigType): The event type.
        """
        super().__init__()
        color = QColor.fromRgbF(*(event_type['box_color']))
        self.rect = map_scene.addRect(0, 0, 16, 16, pen=QPen(0), brush=QBrush(color))
        self.text = map_scene.addText(event_type['name'][0])

        font = QFont('Ubuntu')
        font.setBold(True)
        font.setPixelSize(16)
        self.text.setFont(font)
        self.text.setDefaultTextColor(QColor.fromRgbF(*(event_type['text_color'])))
        self.addToGroup(self.rect)
        self.addToGroup(self.text)

    def alignWithPosition(self, x: int, y: int):
        """Aligns the group with a certain position."""
        self.rect.setPos(x, y)
        self.text.setPos(
            x + 8 - self.text.sceneBoundingRect().width() / 2,
            y + 8 - self.text.sceneBoundingRect().height() / 2,
        )


class EventGroupImage(QGraphicsItemGroup):
    """Subclass for an item group consiting of only a QPixmap."""

    def __init__(
        self,
        map_scene: MapScene,
        image: npt.NDArray[np.int_],
        horizontal_displacement: int,
        vertical_displacement: int,
    ):
        """Initializes the group.

        Args:
            map_scene (MapScene): The map scene.
            image (Image): The image.
            horizontal_displacement (int): The horizontal displacement.
            vertical_displacement (int): The vertical displacement.
        """
        super().__init__()
        self.horizontal_displacement = horizontal_displacement
        self.vertical_displacement = vertical_displacement
        self.pixmap = map_scene.addPixmap(QPixmap.fromImage(ndarray_to_QImage(image)))
        self.addToGroup(self.pixmap)

    def alignWithPosition(self, x: int, y: int):
        """Aligns the group with a certain position."""
        self.pixmap.setPos(
            x + self.horizontal_displacement, y + self.vertical_displacement
        )
