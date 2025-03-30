"""Tab for the blocks of the map widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from pymap.gui import render
from pymap.gui.map.blocks import BlocksScene, BlocksSceneParentMixin

from ..blocks_like import BlocksLikeTab
from .border import BorderScene

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

    from ...map_widget import MapWidget


class BlocksTab(BlocksLikeTab, BlocksSceneParentMixin):
    """Tab for the blocks."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        blocks_widget = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(blocks_widget)

        splitter_selection_and_border = QSplitter(Qt.Orientation.Horizontal)

        splitter_selection_and_border.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        blocks_widget.addWidget(splitter_selection_and_border)

        group_border = QtWidgets.QGroupBox('Border')
        group_border.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        border_layout = QtWidgets.QGridLayout()
        group_border.setLayout(border_layout)
        self.border_scene = BorderScene(self)
        self.border_scene_view = QtWidgets.QGraphicsView()
        self.border_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.border_scene_view.setScene(self.border_scene)
        self.border_scene_view.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        border_layout.addWidget(self.border_scene_view, 1, 1, 1, 1)
        self.show_border = QtWidgets.QCheckBox('Show')
        self.show_border.setChecked(True)
        self.show_border.toggled.connect(self.map_widget.load_map)
        border_layout.addWidget(self.show_border, 2, 1, 1, 1)

        group_selection = QtWidgets.QGroupBox('Selection')
        group_selection_layout = QtWidgets.QGridLayout()
        group_selection.setLayout(group_selection_layout)
        self.selection_scene = QGraphicsScene()
        self.selection_scene_view = QtWidgets.QGraphicsView()
        self.selection_scene_view.setScene(self.selection_scene)
        self.selection_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        group_selection_layout.addWidget(self.selection_scene_view, 1, 1, 2, 1)
        self.select_levels = QtWidgets.QCheckBox('Select Levels')
        self.select_levels.setChecked(False)
        self.select_levels.toggled.connect(self.map_widget.update_layers)
        group_selection_layout.addWidget(self.select_levels, 3, 1, 1, 1)

        splitter_selection_and_border.addWidget(group_selection)
        splitter_selection_and_border.addWidget(group_border)

        group_blocks = QWidget()
        group_blocks_layout = QtWidgets.QGridLayout()
        group_blocks.setLayout(group_blocks_layout)

        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QtWidgets.QGraphicsView()
        self.blocks_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        self.blocks_scene_view.leaveEvent = (
            lambda event: self.map_widget.info_label.clear()
        )
        group_blocks_layout.addWidget(self.blocks_scene_view)
        group_blocks.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        blocks_widget.addWidget(group_blocks)

        blocks_widget.setStretchFactor(0, 0)  # Border
        blocks_widget.setStretchFactor(1, 0)  # Selection and Auto
        blocks_widget.setStretchFactor(2, 1)  # Blocks

        blocks_widget.restoreState(
            self.map_widget.main_gui.settings.value(
                'map_widget/blocks_splitter_state', b'', type=bytes
            )  # type: ignore
        )
        blocks_widget.splitterMoved.connect(
            lambda: self.map_widget.main_gui.settings.setValue(
                'map_widget/blocks_splitter_state', blocks_widget.saveState()
            )
        )
        splitter_selection_and_border.restoreState(
            self.map_widget.main_gui.settings.value(
                'map_widget/selection_and_border_splitter_state', b'', type=bytes
            )  # type: ignore
        )
        splitter_selection_and_border.splitterMoved.connect(
            lambda: self.map_widget.main_gui.settings.setValue(
                'map_widget/selection_and_border_splitter_state',
                splitter_selection_and_border.saveState(),
            )
        )

    @property
    def connectivity_layer(self) -> int:
        """Returns the connectivity layer."""
        return 0

    @property
    def selected_layers(self) -> NDArray[np.uint8]:
        """Returns the selected layers."""
        if self.select_levels.isChecked():
            return np.array([0, 1])
        else:
            return np.array([0])

    def load_header(self):
        """Loads the tab."""
        super().load_header()
        self.load_blocks()
        self.load_border()
        self.show_border.setEnabled(self.map_widget.header_loaded)
        self.select_levels.setEnabled(self.map_widget.header_loaded)

    def load_blocks(self):
        """Loads the block pool."""
        self.blocks_scene.clear()
        if not self.map_widget.main_gui.footer_loaded:
            return
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'

        self.blocks_image = QPixmap.fromImage(
            render.ndarray_to_QImage(render.draw_blocks(map_blocks))
        )
        item = QGraphicsPixmapItem(self.blocks_image)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        item.hoverLeaveEvent = lambda event: self.set_info_text('')

    def load_border(self):
        """Loads the border."""
        self.border_scene.clear()
        if not self.map_widget.main_gui.footer_loaded:
            return
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        border_blocks = self.map_widget.main_gui.get_borders()
        self.border_image = QPixmap.fromImage(
            render.ndarray_to_QImage(
                render.draw_blocks(map_blocks, border_blocks[..., 0])
            )
        )
        self.border_scene.addPixmap(self.border_image)
        self.border_scene.setSceneRect(
            0, 0, border_blocks.shape[1] * 16, border_blocks.shape[0] * 16
        )

    def set_selection(self, selection: NDArray[np.uint8]):
        """Sets the selection.

        Args:
            selection (NDArray[np.uint8]): The selection.
        """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        if not self.map_widget.header_loaded:
            return
        # Block selection
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        selection_pixmap = QPixmap.fromImage(
            render.ndarray_to_QImage(
                render.draw_blocks(map_blocks, self.selection[..., 0])
            )
        )
        item = QGraphicsPixmapItem(selection_pixmap)
        self.selection_scene.addItem(item)
        self.selection_scene.setSceneRect(
            0, 0, selection_pixmap.width(), selection_pixmap.height()
        )

    def load_map(self):
        """Reloads the map image by using tiles of the map widget."""
        self.map_widget.add_block_images_to_scene()

    @property
    def main_gui(self) -> PymapGui:
        """Returns whether the header is loaded."""
        return self.map_widget.main_gui

    def set_info_text(self, text: str):
        """Sets the text of the info label.

        Args:
            text (str): The text to set.
        """
        self.map_widget.info_label.setText(text)
