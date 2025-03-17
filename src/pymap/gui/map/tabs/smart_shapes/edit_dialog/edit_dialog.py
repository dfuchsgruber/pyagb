"""Dialog for editing smart shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QKeyEvent, QKeySequence, QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from agb.model.type import IntArray
from pymap.gui import render
from pymap.gui.history.smart_shape import SetSmartShapeTemplateBlocks
from pymap.gui.map.blocks import BlocksScene, BlocksSceneParentMixin
from pymap.gui.map.tabs.smart_shapes.shape_block_image import (
    smart_shape_get_block_image,
)

from .shape_scene import ShapeScene

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

    from ..smart_shapes import SmartShapesTab


class EditSmartShapeDialog(QDialog, BlocksSceneParentMixin):
    """Dialog for creating smart shapes."""

    def __init__(self, parent: SmartShapesTab, name: str):
        """Initialize the dialog.

        Args:
            parent (SmartShapesTab): The parent tab.
            name (str): The name of the smart shape to edit.
        """
        super().__init__(parent=parent)
        self.smart_shapes_tab = parent

        self.setWindowTitle(f'Edit Smart Shape {name}')
        layout = QGridLayout()
        self.setLayout(layout)

        group_selection = QGroupBox('Selection')
        group_selection_layout = QVBoxLayout()
        group_selection.setLayout(group_selection_layout)
        layout.addWidget(group_selection, 1, 1, 1, 1)
        self.selection_scene = QGraphicsScene()
        self.selection_scene_view = QGraphicsView()
        self.selection_scene_view.setViewport(QOpenGLWidget())
        self.selection_scene_view.setScene(self.selection_scene)
        group_selection_layout.addWidget(self.selection_scene_view)

        group_shape = QGroupBox('Shape')
        group_shape_layout = QVBoxLayout()
        group_shape.setLayout(group_shape_layout)
        layout.addWidget(group_shape, 1, 2, 1, 1)
        self.shape_scene = ShapeScene(self)
        self.shape_scene_view = QGraphicsView()
        self.shape_scene_view.setViewport(QOpenGLWidget())
        self.shape_scene_view.setScene(self.shape_scene)
        group_shape_layout.addWidget(self.shape_scene_view)

        group_blocks = QGroupBox('Blocks')  #
        group_blocks_layout = QVBoxLayout()
        group_blocks.setLayout(group_blocks_layout)
        layout.addWidget(group_blocks, 2, 1, 1, 2)
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QGraphicsView()
        self.blocks_scene_view.setViewport(QOpenGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        group_blocks_layout.addWidget(self.blocks_scene_view)

        self.info_label = QLabel('')
        layout.addWidget(self.info_label, 3, 1, 1, 2)

        self.load_blocks()
        self.shape_scene.load_shape(
            self.smart_shapes_tab.combo_box_smart_shapes.currentText()
        )
        self.update_shape_block_images()

        undo_action = self.smart_shapes_tab.map_widget.undo_stack.createUndoAction(self)
        undo_action.setShortcut(
            QKeySequence.StandardKey.Undo
        )  # This sets Ctrl+Z as the shortcut

    @property
    def main_gui(self) -> PymapGui:
        """The main ui for pymap."""
        return self.smart_shapes_tab.map_widget.main_gui

    def set_selection(self, selection: IntArray) -> None:
        """Sets the selection according to what is selected in this scene.

        Args:
            selection (IntArray): The selection array.
        """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        if not self.main_gui.header_loaded:
            return
        # Block selection
        map_blocks = self.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        selection_pixmap = QPixmap.fromImage(
            ImageQt(render.draw_blocks(map_blocks, self.selection))
        )
        item = QGraphicsPixmapItem(selection_pixmap)
        self.selection_scene.addItem(item)
        self.selection_scene.setSceneRect(
            0, 0, selection_pixmap.width(), selection_pixmap.height()
        )

    def set_info_text(self, text: str) -> None:
        """Sets the text of the info label.

        Args:
            text (str): The text to set.
        """
        self.info_label.setText(text)

    def update_shape_block_images(self):
        """Recomputes the block images for the shape."""
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        smart_shape = self.smart_shapes_tab.current_smart_shape
        assert smart_shape is not None, 'Smart shape is not loaded'
        self.shape_block_images = smart_shape_get_block_image(
            smart_shape,
            self.main_gui.block_images,
        )
        for _, item in np.ndenumerate(self.shape_block_images):
            item.setAcceptHoverEvents(True)
            self.shape_scene.addItem(item)

    def update_shape_with_blocks(self, x: int, y: int, blocks: IntArray) -> None:
        """Updates the shape with a block rectangle at a certain position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            blocks (IntArray): The blocks.
        """
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        for (yy, xx), block_idx in np.ndenumerate(blocks):
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.block_images[block_idx]))
            self.shape_block_images[yy + y, xx + x].setPixmap(pixmap)

    def set_shape_blocks(self, x: int, y: int, blocks: IntArray) -> None:
        """Sets the blocks of the shape.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            blocks (IntArray): The blocks.
        """
        smart_shape = self.smart_shapes_tab.current_smart_shape
        assert smart_shape is not None, 'Smart shape is not loaded'
        assert self.smart_shapes_tab.current_smart_shape_name is not None, (
            'Smart shape name is not loaded'
        )

        # Truncate to fit the smart shape map
        window = smart_shape.blocks[
            y : y + blocks.shape[0], x : x + blocks.shape[1]
        ].copy()
        blocks = blocks[: window.shape[0], : window.shape[1]].copy()

        self.smart_shapes_tab.map_widget.undo_stack.push(
            SetSmartShapeTemplateBlocks(
                self,
                self.smart_shapes_tab.current_smart_shape_name,
                x,
                y,
                blocks,
                window,
            )
        )

    def keyPressEvent(self, arg__1: QKeyEvent):
        """Key press event."""
        if arg__1.matches(QKeySequence.StandardKey.Undo):
            self.smart_shapes_tab.map_widget.undo_stack.undo()
        elif arg__1.matches(QKeySequence.StandardKey.Redo):
            self.smart_shapes_tab.map_widget.undo_stack.redo()
        else:
            super().keyPressEvent(arg__1)
