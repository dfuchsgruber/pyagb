"""Widget for the "blocks" that can be used for level mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

from .child import MapChildMixin, if_header_loaded

if TYPE_CHECKING:
    from .map_widget import MapWidget


class LevelBlocksScene(QGraphicsScene, MapChildMixin):
    """Scene for the blocks level view."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initializes the blocks level scene.

        Args:
            map_widget (MapChildMixin): The map widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        MapChildMixin.__init__(self, map_widget)

    @if_header_loaded
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        pos = event.scenePos()
        x, y = int(pos.x() / 32), int(pos.y() / 32)
        if x < 0 or x >= 4 or y < 0 or y >= 16:
            return self.map_widget.info_label.setText('')
        else:
            self.map_widget.info_label.setText(level_to_info(4 * y + x))

    @if_header_loaded
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        pos = event.scenePos()
        x, y = int(pos.x() / 32), int(pos.y() / 32)
        level = 4 * y + x
        if (
            x in range(4)
            and y in range(16)
            and (
                event.button() == Qt.MouseButton.LeftButton
                or event.button() == Qt.MouseButton.RightButton
            )
        ):
            self.map_widget.set_selection(np.array([[[0, level]]]))


def level_to_info(level: int) -> str:
    """Converts a level to a string with information.

    Args:
        level (int): The level.

    Returns:
        str: The information.
    """
    x, y = level % 4, level // 4

    x_to_collision = {0: 'Passable', 1: 'Obstacle', 2: '??? (2)', 3: '??? (3)'}

    match y:
        case int() if 2 < y < 15:
            return f'Level {hex(y)}, {x_to_collision[x]}'
        case 0:
            x_to_collision = {
                0: 'Connect Levels',
                1: 'Obstacle',
                2: '??? (2)',
                3: '??? (3)',
            }
            return f'{x_to_collision[x]}'
        case 1:
            return f'Water, {x_to_collision[x]}'
        case 15:
            return f'Bridge, {x_to_collision[x]}'
        case _:
            return f'??? (y={y})'
