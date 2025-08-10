"""Dialog for editing smart shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QKeyEvent, QKeySequence, QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from pymap.gui import render
from pymap.gui.history.smart_shape import SetSmartShapeTemplateBlocks
from pymap.gui.map.blocks import BlocksScene, BlocksSceneParentMixin
from pymap.gui.rgba_image import QRGBAImage
from pymap.gui.transparent.view import QGraphicsViewWithTransparentBackground
from pymap.gui.types import Tilemap

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
        self.selection_scene_view = QGraphicsViewWithTransparentBackground()
        self.selection_scene_view.setViewport(QOpenGLWidget())
        self.selection_scene_view.setMouseTracking(True)
        self.selection_scene_view.setScene(self.selection_scene)
        group_selection_layout.addWidget(self.selection_scene_view)

        group_shape = QGroupBox('Shape')
        group_shape_layout = QVBoxLayout()
        group_shape.setLayout(group_shape_layout)
        layout.addWidget(group_shape, 1, 2, 1, 1)
        self.shape_scene = ShapeScene(self)
        self.shape_scene_view = QGraphicsViewWithTransparentBackground()
        self.shape_scene_view.setViewport(QOpenGLWidget())
        self.shape_scene_view.setScene(self.shape_scene)
        group_shape_layout.addWidget(self.shape_scene_view)

        group_blocks = QGroupBox('Blocks')  #
        group_blocks_layout = QVBoxLayout()
        group_blocks.setLayout(group_blocks_layout)
        layout.addWidget(group_blocks, 2, 1, 1, 2)
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QGraphicsViewWithTransparentBackground()
        self.blocks_scene_view.setViewport(QOpenGLWidget())
        self.blocks_scene_view.setMouseTracking(True)
        self.blocks_scene_view.setScene(self.blocks_scene)
        group_blocks_layout.addWidget(self.blocks_scene_view)

        self.info_label = QLabel('')
        layout.addWidget(self.info_label, 3, 1, 1, 2)

        self.load_blocks()
        self.load_shape(self.smart_shapes_tab.combo_box_smart_shapes.currentText())

        undo_action = self.smart_shapes_tab.map_widget.undo_stack.createUndoAction(self)
        undo_action.setShortcut(
            QKeySequence.StandardKey.Undo
        )  # This sets Ctrl+Z as the shortcut

    @property
    def main_gui(self) -> PymapGui:
        """The main ui for pymap."""
        return self.smart_shapes_tab.map_widget.main_gui

    def load_shape(self, shape: str) -> None:
        """Loads the shape.

        Args:
            shape (str): The shape.
        """
        assert self.main_gui.project is not None, 'Project is not loaded'
        smart_shape = self.main_gui.smart_shapes[shape]
        assert self.main_gui.project is not None, 'Project is not loaded'
        template = self.main_gui.project.smart_shape_templates[smart_shape.template]
        map_blocks = self.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        self.shape_scene.addItem(
            self.shape_scene_view.get_transparent_background(
                template.template_qrgba_image.width,
                template.template_qrgba_image.height,
            )
        )
        self.shape_scene.addItem(template.template_qrgba_image.item)
        blocks_image = render.draw_blocks(map_blocks, smart_shape.blocks)
        self.shape_blocks_qrgba_image = QRGBAImage(blocks_image)
        self.shape_blocks_qrgba_image.item.setCacheMode(
            QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        self.shape_blocks_qrgba_image.item.setOpacity(template.blocks_opacity)
        self.shape_scene.addItem(self.shape_blocks_qrgba_image.item)
        self.shape_scene.setSceneRect(
            0,
            0,
            template.template_qrgba_image.width,
            template.template_qrgba_image.height,
        )

    def load_blocks(self):
        """Loads the block pool."""
        self.blocks_scene.clear()
        if not self.smart_shapes_tab.map_widget.main_gui.footer_loaded:
            return
        map_blocks = self.smart_shapes_tab.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        self.blocks_scene.addItem(
            self.blocks_scene_view.get_transparent_background(
                render.blocks_pool.shape[1] * 16, render.blocks_pool.shape[0] * 16
            )
        )
        self.blocks_qrgba_image = QRGBAImage(
            render.draw_blocks(map_blocks, render.blocks_pool[..., 0])
        )

        self.blocks_qrgba_image.item.setCacheMode(
            QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        self.blocks_qrgba_image.item.setAcceptHoverEvents(False)
        self.blocks_scene.addItem(self.blocks_qrgba_image.item)
        self.blocks_scene.setSceneRect(
            0, 0, render.blocks_pool.shape[1] * 16, render.blocks_pool.shape[0] * 16
        )
        self.blocks_qrgba_image.item.hoverLeaveEvent = lambda event: self.set_info_text(
            ''
        )

    def set_selection(self, selection: Tilemap) -> None:
        """Sets the selection according to what is selected in this scene.

        Args:
            selection (Tilemap): The selection array.
        """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        self.selection_scene.addItem(
            self.selection_scene_view.get_transparent_background(
                selection.shape[1] * 16, selection.shape[0] * 16
            )
        )
        if not self.main_gui.header_loaded:
            return
        # Block selection
        map_blocks = self.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        selection_pixmap = QPixmap.fromImage(
            render.ndarray_to_QImage(
                render.draw_blocks(map_blocks, self.selection[..., 0])
            )
        )
        item = QGraphicsPixmapItem(selection_pixmap)
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
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

    def update_shape_with_blocks(self, x: int, y: int, blocks: Tilemap) -> None:
        """Updates the shape with a block rectangle at a certain position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            blocks (Tilemap): The blocks.
        """
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'

        rgba_image = render.draw_blocks(self.main_gui.block_images, blocks)
        self.shape_blocks_qrgba_image.set_rectangle(rgba_image, x * 16, y * 16)

    def set_shape_blocks(self, x: int, y: int, blocks: Tilemap) -> None:
        """Sets the blocks of the shape.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            blocks (Tilemap): The blocks.
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
