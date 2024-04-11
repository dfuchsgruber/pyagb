"""Widget for the auto-shape generation."""

from __future__ import annotations

import importlib.resources as resources
from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageQt import ImageQt
from pymap.gui.render import draw_blocks
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

if TYPE_CHECKING:
    from .auto_shapes import AutoShapesTab


class AutoShapesScene(QGraphicsScene):
    """Scene for automatic shapes."""

    def __init__(self, auto_shapes_tab: AutoShapesTab, parent: QWidget | None = None):
        """Initializes the scene.

        Args:
            auto_shapes_tab (AutoShapesTab): The parent tab.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.auto_shapes_tab = auto_shapes_tab
        self.auto_shape = np.zeros((3, 5, 2), dtype=int)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.auto_shapes_tab.map_widget.header_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(self.auto_shape.shape[1]) and y in range(
            self.auto_shape.shape[0]
        ):
            block_idx = self.auto_shape[y, x, 0]
            self.auto_shapes_tab.map_widget.info_label.setText(
                f'Block : {hex(block_idx)}'
            )

        else:
            return self.auto_shapes_tab.map_widget.info_label.setText('')

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.auto_shapes_tab.map_widget.header_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if (
            x in range(self.auto_shape.shape[1])
            and y in range(self.auto_shape.shape[0])
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Set a new auto-shape
            blocks = self.auto_shapes_tab.map_widget.selection
            assert blocks is not None, 'Expected blocks to be set'
            window = self.auto_shape[
                y : y + blocks.shape[0], x : x + blocks.shape[1]
            ].copy()

            blocks = blocks[: window.shape[0], : window.shape[1]].copy()
            self.auto_shape[
                y : y + blocks.shape[0], x : x + blocks.shape[1], 0
            ] = blocks[:, :, 0]
            self.update_pixmap()

    def update_pixmap(self):
        """Updates the picture of the auto shape."""
        self.clear()

        self.auto_shape_background_pixmap = QPixmap(
            str(
                resources.files('pymap.gui.map.tabs').joinpath(
                    'auto_shape_background.png'
                )
            ),
        )
        self.addPixmap(self.auto_shape_background_pixmap)

        if not self.auto_shapes_tab.map_widget.header_loaded:
            return

        assert (
            self.auto_shapes_tab.map_widget.main_gui.block_images is not None
        ), 'Blocks is None'
        self.blocks_image = QPixmap.fromImage(
            ImageQt(
                draw_blocks(
                    self.auto_shapes_tab.map_widget.main_gui.block_images,
                    self.auto_shape,
                )
            )
        )

        item = QGraphicsPixmapItem(self.blocks_image)
        self.addItem(item)
        item.setAcceptHoverEvents(True)
        self.setSceneRect(
            0, 0, self.auto_shape.shape[1] * 16, self.auto_shape.shape[0] * 16
        )
