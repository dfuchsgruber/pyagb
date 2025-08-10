"""Widget for the "blocks" that can be used for smart shape mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

if TYPE_CHECKING:
    from .smart_shapes import SmartShapesTab


class SmartShapesBlocksScene(QGraphicsScene):
    """Scene for the blocks of the smart shape view."""

    def __init__(self, smart_shapes_tab: SmartShapesTab, parent: QWidget | None = None):
        """Initializes the blocks level scene.

        Args:
            smart_shapes_tab (LevelsTab): The tab widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.smart_shapes_tab = smart_shapes_tab

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.smart_shapes_tab.map_widget.header_loaded:
            return
        assert self.smart_shapes_tab.map_widget.main_gui.project is not None, (
            'Project is not loaded'
        )
        assert self.smart_shapes_tab.current_smart_shape is not None
        template = (
            self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates[
                self.smart_shapes_tab.current_smart_shape.template
            ]
        )
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        block_num = self.smart_shapes_tab.num_smart_shape_blocks_per_row * y + x
        if (
            0 <= block_num < template.num_blocks
            and 0 <= x < self.smart_shapes_tab.num_smart_shape_blocks_per_row
        ):
            self.smart_shapes_tab.map_widget.info_label.setText(
                template.block_tooltips[block_num]
            )
        else:
            self.smart_shapes_tab.map_widget.info_label.setText('')

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.smart_shapes_tab.map_widget.header_loaded:
            return
        assert self.smart_shapes_tab.map_widget.main_gui.project is not None, (
            'Project is not loaded'
        )
        assert self.smart_shapes_tab.current_smart_shape is not None
        template = (
            self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates[
                self.smart_shapes_tab.current_smart_shape.template
            ]
        )
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        block_num = self.smart_shapes_tab.num_smart_shape_blocks_per_row * y + x
        if (
            0 <= block_num < template.num_blocks
            and (
                event.button() == Qt.MouseButton.LeftButton
                or event.button() == Qt.MouseButton.RightButton
            )
            and 0 <= x < self.smart_shapes_tab.num_smart_shape_blocks_per_row
        ):
            self.smart_shapes_tab.set_selection(np.array([[[block_num, 0]]]))
