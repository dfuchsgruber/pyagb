"""Tab for the blocks of the map widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QSizePolicy,
    QSplitter,
    QWidget,
)

from pymap.gui import render
from pymap.gui.map.blocks import BlocksScene, BlocksSceneParentMixin
from pymap.gui.map.view import VisibleLayer
from pymap.gui.rgba_image import QRGBAImage
from pymap.gui.transparent.view import QGraphicsViewWithTransparentBackground
from pymap.gui.types import MapLayers, Tilemap

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
        self.border_scene_view = QGraphicsViewWithTransparentBackground()
        self.border_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.border_scene_view.setMouseTracking(True)
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
        self.selection_scene_view = QGraphicsViewWithTransparentBackground()
        self.selection_scene_view.setScene(self.selection_scene)
        self.selection_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.selection_scene_view.setMouseTracking(True)
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
        self.blocks_scene_view = QGraphicsViewWithTransparentBackground()
        self.blocks_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.blocks_scene_view.setMouseTracking(True)
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
    def visible_layers(self) -> VisibleLayer:
        """Get the visible layers."""
        return VisibleLayer.BLOCKS | VisibleLayer.BORDER_EFFECT | VisibleLayer.GRID

    @property
    def selected_layers(self) -> MapLayers:
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

    def load_border(self):
        """Loads the border."""
        self.border_scene.clear()
        if not self.map_widget.main_gui.footer_loaded:
            return
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        border_blocks = self.map_widget.main_gui.get_borders()

        self.border_scene.addItem(
            self.border_scene_view.get_transparent_background(
                border_blocks.shape[1] * 16, border_blocks.shape[0] * 16
            )
        )
        self.border_qrgba_image = QRGBAImage(
            render.draw_blocks(map_blocks, border_blocks[..., 0])
        )
        self.border_qrgba_image.item.setCacheMode(
            QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        self.border_scene.addItem(self.border_qrgba_image.item)
        self.border_scene.setSceneRect(
            0, 0, border_blocks.shape[1] * 16, border_blocks.shape[0] * 16
        )

    def update_border_block_at(self, x: int, y: int):
        """Updates the border block at the given position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        if not self.map_widget.main_gui.footer_loaded:
            return
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        border_blocks = self.map_widget.main_gui.get_borders()
        block_idx = border_blocks[y, x, 0]
        self.border_qrgba_image.set_rectangle(map_blocks[block_idx], 16 * x, 16 * y)

    def update_block_idx(self, block_idx: int):
        """Updates the block with a given index.

        This redraws the block in the pool and in the border if it is shown.

        Args:
            block_idx (int): The index of the block to update.
        """
        super().update_block_idx(block_idx)
        if not self.map_widget.main_gui.footer_loaded:
            return

        assert self.map_widget.main_gui.block_images is not None, (
            'Blocks are not loaded'
        )
        border_blocks = self.map_widget.main_gui.get_borders()
        for y, x in zip(*np.where(border_blocks[..., 0] == block_idx)):
            self.update_border_block_at(x, y)
        x = block_idx % render.blocks_pool.shape[1]
        y = block_idx // render.blocks_pool.shape[1]
        self.blocks_qrgba_image.set_rectangle(
            self.map_widget.main_gui.block_images[block_idx], 16 * x, 16 * y
        )

    def set_selection(self, selection: Tilemap):
        """Sets the selection.

        Args:
            selection (RGBAImage): The selection.
        """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        # Add a transparent background
        self.selection_scene.addItem(
            self.selection_scene_view.get_transparent_background(
                selection.shape[1] * 16, selection.shape[0] * 16
            )
        )
        if not self.map_widget.header_loaded:
            return
        # Block selection
        map_blocks = self.map_widget.main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        selection_qrgba_image = QRGBAImage(
            render.draw_blocks(map_blocks, selection[..., 0])
        )

        selection_qrgba_image.item.setCacheMode(
            QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        self.selection_scene.addItem(selection_qrgba_image.item)
        self.selection_scene.setSceneRect(
            0,
            0,
            selection_qrgba_image.pixmap.width(),
            selection_qrgba_image.pixmap.height(),
        )

    def load_map(self):
        """Reloads the map image by using tiles of the map widget."""

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
