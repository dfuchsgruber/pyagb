"""Widget for the border."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.gui.transparent.scene import QGraphicsSceneWithTransparentBackground

if TYPE_CHECKING:
    from . import BlocksTab


class BorderScene(QGraphicsSceneWithTransparentBackground):
    """Scene for the border view."""

    def __init__(self, map_widget: BlocksTab, parent: QWidget | None = None):
        """Initializes the border scene.

        Args:
            map_widget (MapWidget): The map widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.blocks_tab = map_widget

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.blocks_tab.map_widget.header_loaded:
            return
        assert self.blocks_tab.map_widget.main_gui.project is not None, (
            'Project is not loaded'
        )
        borders = self.blocks_tab.map_widget.main_gui.get_borders()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(borders.shape[1]) and y in range(borders.shape[0]):
            block_idx = borders[y, x, 0]
            self.blocks_tab.map_widget.info_label.setText(f'Block : {hex(block_idx)}')
        else:
            return self.blocks_tab.map_widget.info_label.setText('')

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.blocks_tab.map_widget.header_loaded:
            return
        borders = self.blocks_tab.map_widget.main_gui.get_borders()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if (
            x in range(borders.shape[1])
            and y in range(borders.shape[0])
            and event.button() == Qt.MouseButton.LeftButton
        ):
            assert self.blocks_tab.selection is not None, 'Selection is not set'
            self.blocks_tab.map_widget.main_gui.set_border_at(
                x, y, self.blocks_tab.selection
            )
