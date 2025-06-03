"""Widget for the "blocks" that can be used for smart shape mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
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

    @property
    def num_smart_shape_blocks_per_row(self) -> int:
        """The number of smart shape blocks per row."""  #
        assert self.smart_shapes_tab.map_widget.main_gui.project is not None, (
            'Project is not loaded'
        )
        return self.smart_shapes_tab.map_widget.main_gui.project.config['pymap'][
            'display'
        ]['smart_shape_blocks_per_row']

    def load_smart_shape(self):
        """Updates the blocks scene with the current smart shape."""
        self.clear()
        if self.smart_shapes_tab.current_smart_shape is not None:
            assert self.smart_shapes_tab.map_widget.main_gui.project is not None, (
                'Project is not loaded'
            )
            template = (
                self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates[
                    self.smart_shapes_tab.current_smart_shape.template
                ]
            )
            cols = self.num_smart_shape_blocks_per_row
            for idx, (pixmap, tooltip) in enumerate(
                zip(template.block_pixmaps, template.block_tooltips)
            ):
                item = QGraphicsPixmapItem(pixmap)
                item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
                x, y = idx % cols, idx // cols
                item.setPos(16 * x, 16 * y)
                item.setAcceptHoverEvents(True)
                item.setToolTip(tooltip)
                self.addItem(item)

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
        block_num = self.num_smart_shape_blocks_per_row * y + x
        if (
            0 <= block_num < template.num_blocks
            and 0 <= x < self.num_smart_shape_blocks_per_row
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
        block_num = self.num_smart_shape_blocks_per_row * y + x
        if (
            0 <= block_num < template.num_blocks
            and (
                event.button() == Qt.MouseButton.LeftButton
                or event.button() == Qt.MouseButton.RightButton
            )
            and 0 <= x < self.num_smart_shape_blocks_per_row
        ):
            self.smart_shapes_tab.set_selection(np.array([[[block_num, 0]]]))
