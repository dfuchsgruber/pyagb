"""Widget that shows the map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsScene,
    QWidget,
)

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


class MapScene(QGraphicsScene):
    """Scene that will show the map."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the map scene.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.grid_group = None

    def clear(self) -> None:
        """Clears the scene."""
        super().clear()
        self.grid_group = None

    def update_grid(self):
        """Updates the grid of the scene."""
        if self.grid_group is not None:
            self.removeItem(self.grid_group)
        if self.main_gui.grid_visible:
            self.grid_group = QGraphicsItemGroup()
            for x in range(0, int(self.width()), 16):
                self.grid_group.addToGroup(self.addLine(x, 0, x, self.height()))
            for y in range(0, int(self.height()), 16):
                self.grid_group.addToGroup(self.addLine(0, y, self.width(), y))
            self.addItem(self.grid_group)
        else:
            self.grid_group = None
