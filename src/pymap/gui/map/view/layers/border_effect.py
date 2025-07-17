"""Handles the effect over the borders in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
)

from .layer import MapViewLayer

if TYPE_CHECKING:
    pass


from PySide6.QtCore import Qt


class MapViewLayerBorderEffects(MapViewLayer):
    """A layer in the map view that displays border effects."""

    def load_map(self) -> None:
        """Loads the border effects into the scene."""
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        padded_width, padded_height = self.view.main_gui.get_border_padding()
        map_width, map_height = self.view.main_gui.get_map_dimensions()

        # Apply shading to border parts by adding opaque rectangles
        border_color = QColor.fromRgbF(
            *(self.view.main_gui.project.config['pymap']['display']['border_color'])
        )
        self.item = QGraphicsItemGroup()
        brush = QBrush(border_color)
        no_border_pen = QPen(Qt.PenStyle.NoPen)
        # no outline
        self.north_border = QGraphicsRectItem(
            0,
            0,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
        )
        self.north_border.setBrush(brush)
        self.north_border.setPen(no_border_pen)
        self.south_border = QGraphicsRectItem(
            0,
            (padded_height + map_height) * 16,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
        )
        self.south_border.setBrush(brush)
        self.south_border.setPen(no_border_pen)
        self.west_border = QGraphicsRectItem(
            0,
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
        )
        self.west_border.setPen(no_border_pen)
        self.west_border.setBrush(brush)
        self.east_border = QGraphicsRectItem(
            16 * (padded_width + map_width),
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
        )
        self.east_border.setBrush(brush)
        self.east_border.setPen(no_border_pen)
        self.item.addToGroup(self.north_border)
        self.item.addToGroup(self.south_border)
        self.item.addToGroup(self.west_border)
        self.item.addToGroup(self.east_border)
