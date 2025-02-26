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

from pymap.gui.map.level import level_to_info

if TYPE_CHECKING:
    from .levels import LevelsTab


class LevelBlocksScene(QGraphicsScene):
    """Scene for the blocks level view."""

    def __init__(self, levels_tab: LevelsTab, parent: QWidget | None = None):
        """Initializes the blocks level scene.

        Args:
            levels_tab (LevelsTab): The tab widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.levels_tab = levels_tab

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.levels_tab.map_widget.header_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 32), int(pos.y() / 32)
        if x < 0 or x >= 4 or y < 0 or y >= 16:
            return self.levels_tab.map_widget.info_label.setText('')
        else:
            self.levels_tab.map_widget.info_label.setText(level_to_info(4 * y + x))

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.levels_tab.map_widget.header_loaded:
            return
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
            self.levels_tab.set_selection(np.array([[[0, level]]]))
