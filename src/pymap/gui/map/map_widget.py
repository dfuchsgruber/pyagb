"""Widget for the map."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QWidget,
)

from pymap.gui import blocks
from pymap.gui.blocks import compute_blocks
from pymap.gui.render import ndarray_to_QImage
from pymap.gui.types import MapLayers, Tilemap

from .map_scene import MapScene
from .tabs.blocks import BlocksTab
from .tabs.levels import LevelsTab
from .tabs.smart_shapes.smart_shapes import SmartShapesTab
from .tabs.tab import MapWidgetTab

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class MapWidgetTabType(IntEnum):
    """Enum for the map widget tabs."""

    BLOCKS = 0
    LEVELS = 1
    AUTO_SHAPES = 2


class MapTabsWidget(QTabWidget):
    """Widget for the map tabs."""

    def currentWidget(self) -> MapWidgetTab:
        """Get the current widget.

        Returns:
            MapWidgetTab: The current widget.
        """
        widget = super().currentWidget()
        assert isinstance(widget, MapWidgetTab), 'Widget is not a MapWidgetTab'
        return widget

    def widget(self, index: int) -> MapWidgetTab:
        """Get the widget at a certain index.

        Args:
            index (int): The index.

        Returns:
            MapWidgetTab: The widget.
        """
        widget = super().widget(index)
        assert isinstance(widget, MapWidgetTab), 'Widget is not a MapWidgetTab'
        return widget


class MapWidget(QWidget):
    """Widget for the map and its properties."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the map widget.

        Args:
            main_gui (PymapGui): The main gui.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        # Store blocks in an seperate numpy array that contains the border as well
        self.blocks = None  # Array of map blocks *with* padding
        self.layers = np.array(0)
        self.undo_stack = QtGui.QUndoStack()
        self.undo_stack.canRedoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.canUndoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.indexChanged.connect(self._update_undo_redo_tooltips)

        # (Re-)Build the ui
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        layout.addWidget(splitter, 1, 1, 1, 1)

        self.map_scene = MapScene(self)
        self.map_scene_view = QtWidgets.QGraphicsView()
        self.map_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.map_scene_view.setScene(self.map_scene)
        self.map_scene_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.map_scene_view.leaveEvent = lambda event: self.info_label.clear()
        splitter.addWidget(self.map_scene_view)

        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label, 2, 1, 1, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)

        # Divide into a widget for blocks and levels
        self.tabs = MapTabsWidget()

        self.blocks_tab = BlocksTab(self)
        self.tabs.insertTab(MapWidgetTabType.BLOCKS, self.blocks_tab, 'Blocks')
        self.levels_tab = LevelsTab(self)
        self.tabs.insertTab(MapWidgetTabType.LEVELS, self.levels_tab, 'Level')
        self.smart_shapes_tab = SmartShapesTab(self)
        self.tabs.insertTab(
            MapWidgetTabType.AUTO_SHAPES, self.smart_shapes_tab, 'Smart Shapes'
        )
        self.tabs.currentChanged.connect(self.tab_changed)

        splitter.addWidget(self.tabs)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        splitter.restoreState(
            self.main_gui.settings.value(
                'map_widget/horizontal_splitter_state', b'', type=bytes
            )  # type: ignore
        )
        splitter.splitterMoved.connect(
            lambda: self.main_gui.settings.setValue(
                'map_widget/horizontal_splitter_state', splitter.saveState()
            )
        )
        self.load_header()  # Disables all widgets

    @property
    def header_loaded(self) -> bool:
        """Whether the header is loaded.

        Returns:
            bool: Whether the header is loaded.
        """
        return self.main_gui.header_loaded

    def _update_undo_redo_tooltips(
        self,
    ):
        """Updates the undo and redo tooltips."""
        self.main_gui.update_redo_undo_tooltips(
            self,
            self.undo_stack,
        )

    # @ProfileBlock('MapWidget:tab_changed')
    def tab_changed(self, *args: Any, **kwargs: Any):
        """Triggered when the user switches from the blocks to levels tab."""
        self.update_layers()
        self.load_map()

    def update_layers(self):
        """Updates the current layers."""
        self.layers = self.tabs.currentWidget().selected_layers

    def load_project(self, *args: Any):
        """Update project related widgets."""
        if not self.main_gui.project_loaded:
            return
        assert self.main_gui.project is not None, 'Project is not loaded'
        for idx in range(self.tabs.count()):
            self.tabs.widget(idx).load_project()
        self.load_header()

    def load_header(self, *args: Any):
        """Updates the entire header related widgets."""
        for idx in range(self.tabs.count()):
            self.tabs.widget(idx).load_header()
        self.load_map()

        if self.main_gui.project is None or self.main_gui.header is None:
            self.blocks = None

    def load_map(self):
        """Loads the entire map image."""
        self.map_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Crop the visible blocks from all blocks including the border
        self.blocks = compute_blocks(self.main_gui.footer, self.main_gui.project)
        connections = self.main_gui.get_connections()
        for connection in blocks.filter_visible_connections(
            blocks.unpack_connections(connections, self.main_gui.project)
        ):
            blocks.insert_connection(
                self.blocks, connection, self.main_gui.footer, self.main_gui.project
            )
        visible_width, visible_height = (
            map_width + 2 * padded_width,
            map_height + 2 * padded_height,
        )
        invisible_border_width, invisible_border_height = (
            (self.blocks.shape[1] - visible_width) // 2,
            (self.blocks.shape[0] - visible_height) // 2,
        )
        self.blocks = self.blocks[
            invisible_border_height : self.blocks.shape[0] - invisible_border_height,
            invisible_border_width : self.blocks.shape[1] - invisible_border_width,
        ]
        self.update_block_images()
        self.update_level_images()
        self.update_smart_shape_images()
        self.tabs.currentWidget().load_map()
        self.update_border_effect()
        self.update_grid()

    def update_grid(self):
        """Updates the grid."""
        self.map_scene.update_grid()

    def update_block_images(self):
        """Recomputes the block images for the map."""
        assert self.blocks is not None, 'Blocks are not loaded'
        self.block_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        for (y, x), block_idx in np.ndenumerate(self.blocks[:, :, 0]):
            # Draw the blocks
            assert self.main_gui.block_images is not None, 'Blocks are not loaded'
            pixmap = QPixmap.fromImage(
                ndarray_to_QImage(self.main_gui.block_images[block_idx])
            )
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            item.setPos(16 * x, 16 * y)
            self.block_images[y, x] = item

    def update_level_images(self):
        """Recomputes the level images for the map."""
        assert self.blocks is not None, 'Blocks are not loaded'
        self.level_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        for (y, x), level in np.ndenumerate(self.blocks[:, :, 1]):
            if x in range(padded_width, padded_width + map_width) and y in range(
                padded_height, padded_height + map_height
            ):
                # Draw the pixmaps
                pixmap = self.levels_tab.level_blocks_pixmaps[level]
                item = QGraphicsPixmapItem(pixmap)
                item.setAcceptHoverEvents(True)
                opacity = QGraphicsOpacityEffect()
                opacity.setOpacity(
                    self.levels_tab.level_opacity_slider.sliderPosition() / 20
                )
                item.setGraphicsEffect(opacity)
                item.setPos(16 * x, 16 * y)
                self.level_images[y, x] = item

    def update_smart_shape_images(self):
        """Updates the images of the smart shape meta block map."""
        assert self.blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.project is not None, 'Project is not loaded'
        self.smart_shape_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        smart_shape = self.smart_shapes_tab.current_smart_shape
        if not smart_shape:
            return

        pixmaps = self.main_gui.project.smart_shape_templates[
            smart_shape.template
        ].block_pixmaps

        for (y, x), _ in np.ndenumerate(self.blocks[:, :, 1]):
            if x in range(padded_width, padded_width + map_width) and y in range(
                padded_height, padded_height + map_height
            ):
                block_idx: int = self.smart_shapes_tab.smart_shape_blocks[
                    y - padded_height, x - padded_width, 0
                ]
                item = QGraphicsPixmapItem(pixmaps[block_idx])
                item.setAcceptHoverEvents(True)
                opacity = QGraphicsOpacityEffect()
                opacity.setOpacity(0.5)
                item.setGraphicsEffect(opacity)
                item.setPos(16 * x, 16 * y)
                self.smart_shape_images[y, x] = item

    def update_border_effect(self):
        """Recomputes the rectangles that create the border opacity effect."""
        assert self.main_gui.project is not None, 'Project is not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Apply shading to border parts by adding opaque rectangles
        border_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['border_color'])
        )
        self.north_border = self.map_scene.addRect(
            0,
            0,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.south_border = self.map_scene.addRect(
            0,
            (padded_height + map_height) * 16,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.west_border = self.map_scene.addRect(
            0,
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.east_border = self.map_scene.addRect(
            16 * (padded_width + map_width),
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.map_scene.setSceneRect(
            0,
            0,
            16 * (2 * padded_width + map_width),
            16 * (2 * padded_height + map_height),
        )

    def update_blocks_at_padded_indices(
        self, indices_padded: tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]
    ):
        """Updates map blocks at certain indices.

        Args:
            indices_padded (tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]):
                The indices.
        """
        assert self.blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        idx_y, idx_x = indices_padded
        for y, x, block_idx in zip(idx_y, idx_x, self.blocks[idx_y, idx_x, 0]):
            y, x, block_idx = cast(int, y), cast(int, x), cast(int, block_idx)
            pixmap = QPixmap.fromImage(
                ndarray_to_QImage(self.main_gui.block_images[block_idx])
            )
            self.block_images[y, x].setPixmap(pixmap)

    def update_layer_at_indices(
        self,
        indices: tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_],
        layer: int,
        indices_padded: bool = False,
    ):
        """Updates the map image at certain indices.

        Args:
            indices (tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]):
                The indices.
            layer (int): The layer.
            indices_padded (bool, optional): Whether the indices are padded.
                Defaults to False.
        """
        indices = np.array(indices).copy()
        if not indices_padded:
            padded_width, padded_height = self.main_gui.get_border_padding()
            indices[0] += padded_height
            indices[1] += padded_width
        if layer == 0:
            self.update_blocks_at_padded_indices(indices)
        elif layer == 1:
            self.update_levels_at_padded_indices(indices)
        else:
            raise ValueError('Invalid layer')

    def update_levels_at_padded_indices(
        self, indices_padded: tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]
    ):
        """Updates the level images at certain indices.

        Args:
            indices_padded (tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]):
                The indices.

        """
        assert self.blocks is not None, 'Blocks are not loaded'
        idx_y, idx_x = indices_padded
        for y, x, level in zip(idx_y, idx_x, self.blocks[idx_y, idx_x, 1]):
            y, x, level = cast(int, y), cast(int, x), cast(int, level)
            self.level_images[y, x].setPixmap(
                self.levels_tab.level_blocks_pixmaps[level]
            )

    def update_map_at_indices(
        self,
        indices: tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_],
        blocks: Tilemap,
        layer: int,
    ):
        """Updates the map image at certain indices.

        Args:
            indices (tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]):
                The indices.
            blocks (Tilemap): The blocks.
            layer (int): The layer.
        """
        if not self.main_gui.footer_loaded:
            return
        assert self.blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        self.blocks[indices[0] + padded_height, indices[1] + padded_width, layer] = (
            blocks[...]
        )
        indices = np.array(indices).copy()
        indices[0] += padded_height
        indices[1] += padded_width
        assert isinstance(self.layers, Iterable), 'Layers is not an iterable'
        if 0 in self.layers:
            self.update_blocks_at_padded_indices(indices)
        if 1 in self.layers:
            self.update_levels_at_padded_indices(indices)

    def update_map_at(self, x: int, y: int, layers: MapLayers, blocks: Tilemap):
        """Updates the map image with new blocks rooted at a certain position."""
        if not self.main_gui.footer_loaded:
            return
        assert self.blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        self.blocks[
            padded_height + y : padded_height + y + blocks.shape[0],
            padded_width + x : padded_width + x + blocks.shape[1],
            layers,
        ] = blocks[:, :, layers]
        assert isinstance(layers, Iterable), 'Layers is not an iterable'
        # Redraw relevant pixel maps
        indices = np.indices((blocks.shape[0], blocks.shape[1])).reshape(2, -1)
        indices[0] += y + padded_height
        indices[1] += x + padded_width
        if 0 in layers:
            self.update_blocks_at_padded_indices(
                indices,  # type: ignore
            )
        if 1 in layers:
            self.update_levels_at_padded_indices(
                indices,  # type: ignore
            )

    def update_map_with_smart_shape_blocks_at(self, x: int, y: int, blocks: Tilemap):
        """Updates the smart shape meta block map at a position with blocks."""
        if not self.main_gui.footer_loaded:
            return
        smart_shape = self.main_gui.smart_shapes[
            self.smart_shapes_tab.current_smart_shape_name
        ]
        assert self.main_gui.project is not None
        template = self.main_gui.project.smart_shape_templates[smart_shape.template]
        smart_shape.buffer[y : y + blocks.shape[0], x : x + blocks.shape[1]] = blocks
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        # Redraw relevant block pixmaps
        for (yy, xx), block_idx in np.ndenumerate(blocks[:, :, 0]):
            pixmap = template.block_pixmaps[block_idx]
            if x + xx in range(map_width) and y + yy in range(map_height):
                self.smart_shape_images[
                    padded_height + y + yy, padded_width + x + xx
                ].setPixmap(pixmap)

    def add_block_images_to_scene(self):
        """Adds the block images to the scene."""
        for (y, x), item in np.ndenumerate(self.block_images[:, :]):
            item.setPos(16 * x, 16 * y)
            self.map_scene.addItem(item)

    def add_level_images_to_scene(self):
        """Adds the level images to the scene."""
        for (y, x), item in np.ndenumerate(self.level_images[:, :]):
            if item is not None:
                item.setPos(16 * x, 16 * y)
                self.map_scene.addItem(item)

    def add_smart_shape_images_to_scene(self):
        """Adds the smart shape images to the scene."""
        for (y, x), item in np.ndenumerate(self.smart_shape_images[:, :]):
            if item is not None:
                item.setPos(16 * x, 16 * y)
                self.map_scene.addItem(item)
