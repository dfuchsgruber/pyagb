"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.gui.map_scene import MapScene

from ..tab import MapWidgetTab

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class EventsTab(MapWidgetTab):
    """Tabs with block like functionality.

    They have a selection of blocks that can be drawn to the map. They also
    support flood filling and replacement. The selection can be taken from the map as
    well or set natively via the `set_selection` method to `selection`.
    """

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

    def load_project(self):
        """Loads the project."""

    def load_header(self):
        """Loads the tab."""

    @property
    def visible_layers(self) -> MapScene.VisibleLayer:
        """Get the visible layers."""
        return (
            MapScene.VisibleLayer.BLOCKS
            | MapScene.VisibleLayer.EVENTS
            | MapScene.VisibleLayer.BORDER_EFFECT
            | MapScene.VisibleLayer.CONNECTIONS
        )

    def map_scene_mouse_pressed(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """

    def map_scene_mouse_moved(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """

    def map_scene_mouse_released(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
