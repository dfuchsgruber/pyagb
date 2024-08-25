"""Widget for the blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

import pymap.gui.render as render

if TYPE_CHECKING:
    from .parent import BlocksSceneParent


class BlocksScene(QGraphicsScene):
    """Scene for the blocks view."""

    def __init__(
        self, blocks_scene_parent: BlocksSceneParent, parent: QWidget | None = None
    ):
        """Initializes the blocks scene.

        Args:
            blocks_scene_parent (BlocksTab): The blocks tab.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.blocks_scene_parent = blocks_scene_parent
        self.selection_box = None

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.blocks_scene_parent.main_gui.footer_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        block_idx = 8 * y + x
        if x < 0 or x >= 8 or y < 0 or y >= 128:
            return self.blocks_scene_parent.set_info_text('')
        else:
            self.blocks_scene_parent.set_info_text(f'Block : {hex(block_idx)}')
        if self.selection_box is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.blocks_scene_parent.set_selection(
                    render.select_blocks(render.blocks_pool, *self.selection_box)
                )

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if not self.blocks_scene_parent.main_gui.footer_loaded:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.selection_box = None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.blocks_scene_parent.main_gui.footer_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if event.button() == Qt.MouseButton.RightButton:
            self.selection_box = x, x + 1, y, y + 1
        if (
            event.button() == Qt.MouseButton.LeftButton
            or event.button() == Qt.MouseButton.RightButton
        ):
            # Select the current block
            self.blocks_scene_parent.set_selection(
                render.blocks_pool[y : y + 1, x : x + 1, :]
            )
