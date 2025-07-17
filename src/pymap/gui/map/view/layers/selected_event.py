"""Handles the the rectangle for the selected event."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymap.configuration import PymapEventConfigType

from .layer import MapViewLayer, VisibleLayer

if TYPE_CHECKING:
    pass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
)

from pymap.gui import properties


class MapViewLayerSelectedEvent(MapViewLayer):
    """A layer in the map view that displays the selected event."""

    def load_map(self) -> None:
        """Loads the selected event into the scene."""
        self.item = QGraphicsItemGroup()
        rect = QGraphicsRectItem(0, 0, 16, 16)
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        rect.setPen(QPen(color, 2.0))
        rect.setBrush(Qt.BrushStyle.NoBrush)
        self.item.addToGroup(rect)

    def update_selected_event_image(
        self, event_type: PymapEventConfigType, event_idx: int | None
    ):
        """Updates the selected event image (the red rectangle).

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index. If None or -1, the event is hidden.
        """
        assert self.view.main_gui.project is not None
        if self.item is None:
            return
        if event_idx is None or event_idx < 0:
            # Hide the selected event group
            self.item.setVisible(False)
            return
        # Show the selected event group
        if self.view.visible_layers & VisibleLayer.SELECTED_EVENT:
            self.item.setVisible(True)

        event = self.view.main_gui.get_event(event_type, event_idx)
        padded_x, padded_y = self.view.main_gui.get_border_padding()
        x, y = self.view.pad_coordinates(
            properties.get_member_by_path(event, event_type['x_path']),
            properties.get_member_by_path(event, event_type['y_path']),
            padded_x,
            padded_y,
        )
        self.item.setPos(16 * x, 16 * y)
