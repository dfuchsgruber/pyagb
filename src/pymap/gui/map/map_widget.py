"""Widget for the map."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QWidget,
)

from pymap.gui.map.map_scene import MapScene
from pymap.gui.types import MapLayers, Tilemap

from .tabs.blocks import BlocksTab
from .tabs.connections import ConnectionsTab
from .tabs.events import EventsTab
from .tabs.levels import LevelsTab
from .tabs.smart_shapes.smart_shapes import SmartShapesTab
from .tabs.tab import MapWidgetTab
from .view import MapView

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class MapWidgetTabType(IntEnum):
    """Enum for the map widget tabs."""

    BLOCKS = 0
    LEVELS = 1
    AUTO_SHAPES = 2
    EVENTS = 3
    CONNECTIONS = 4


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

    THRESHOLD_NUM_BLOCKS_TO_RELOAD_IMAGE = 50

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the map widget.

        Args:
            main_gui (PymapGui): The main gui.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        # Store blocks in an seperate numpy array that contains the border as well
        self.undo_stack = QtGui.QUndoStack()
        self.undo_stack.canRedoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.canUndoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.indexChanged.connect(self._update_undo_redo_tooltips)

        grid_layout = QtWidgets.QGridLayout()
        # (Re-)Build the ui
        splitter = QSplitter(Qt.Orientation.Horizontal)

        grid_layout.addWidget(splitter, 2, 1, 1, 1)

        self.map_scene = MapScene(self)
        self.map_scene_view = MapView(self)
        self.map_scene_view.setScene(self.map_scene)
        self.map_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.map_scene_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.map_scene_view.leaveEvent = lambda event: self.info_label.clear()
        splitter.addWidget(self.map_scene_view)

        self.info_label = QtWidgets.QLabel()
        grid_layout.addWidget(self.info_label, 3, 1, 1, 1)
        grid_layout.setRowStretch(2, 1)
        grid_layout.setRowStretch(3, 0)

        # Divide into a widget for blocks and levels
        self.tabs = MapTabsWidget()

        self.blocks_tab = BlocksTab(self)
        self.tabs.insertTab(MapWidgetTabType.BLOCKS, self.blocks_tab, '&Blocks')
        self.levels_tab = LevelsTab(self)
        self.tabs.insertTab(MapWidgetTabType.LEVELS, self.levels_tab, '&Level')
        self.smart_shapes_tab = SmartShapesTab(self)
        self.tabs.insertTab(
            MapWidgetTabType.AUTO_SHAPES, self.smart_shapes_tab, '&Smart Shapes'
        )
        self.events_tab = EventsTab(self)
        self.tabs.insertTab(MapWidgetTabType.EVENTS, self.events_tab, '&Events')
        self.connections_tab = ConnectionsTab(self)
        self.tabs.insertTab(
            MapWidgetTabType.CONNECTIONS, self.connections_tab, '&Connections'
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
        self.setLayout(grid_layout)
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

    # @Profile('MapWidget:tab_changed')
    def tab_changed(self, *args: Any, **kwargs: Any):
        """Triggered when the user switches from the blocks to levels tab."""
        widget = self.tabs.currentWidget()
        self.map_scene_view.update_visible_layers(widget.visible_layers)

    def update_layers(self):
        """Updates the current layers."""
        # self.layers = self.tabs.currentWidget().selected_layers

    def load_project(self, *args: Any):
        """Update project related widgets."""
        if not self.main_gui.project_loaded:
            return
        assert self.main_gui.project is not None, 'Project is not loaded'
        for idx in range(self.tabs.count()):
            self.tabs.widget(idx).load_project()
        self.map_scene_view.load_project()
        self.load_header()

    def load_header(self, *args: Any):
        """Updates the entire header related widgets."""
        self.load_map()
        # The tabs are loaded after the map scene is built, because some of them
        # move items in the scene (e.g. the selection rectangle for events)
        for idx in range(self.tabs.count()):
            self.tabs.widget(idx).load_header()

    def load_map(self):
        """Loads the entire map image."""
        self.map_scene_view.scene().clear()
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.map_scene_view.load_map()
        self.map_scene_view.update_visible_layers(
            self.tabs.currentWidget().visible_layers
        )

    def update_grid(self):
        """Updates the grid."""
        self.map_scene_view.update_grid()

    def update_blocks(self):
        """Re-computes all blocks and updates the images for blocks that changed."""
        assert self.map_scene_view.visible_blocks is not None
        blocks_previous = self.map_scene_view.visible_blocks.copy()
        self.map_scene_view.compute_visible_blocks()
        self.update_blocks_at_padded_indices(
            np.where(
                blocks_previous[..., 0] != self.map_scene_view.visible_blocks[..., 0]
            )
        )

    def update_block_idx(self, block_idx: int):
        """Updates all blocks with a certain block index."""
        assert self.map_scene_view.visible_blocks is not None, 'Blocks are not loaded'
        self.update_blocks_at_padded_indices(
            np.where(self.map_scene_view.visible_blocks[..., 0] == block_idx)
        )
        for idx in range(self.tabs.count()):
            self.tabs.widget(idx).update_block_idx(block_idx)

    def update_blocks_at_padded_indices(
        self, indices_padded: tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]
    ):
        """Updates map blocks at certain indices.

        Args:
            indices_padded (tuple[npt.NDArray[np.int_], ...] | npt.NDArray[np.int_]):
                The indices.
        """
        idx_y, idx_x = indices_padded
        if len(idx_y) > self.THRESHOLD_NUM_BLOCKS_TO_RELOAD_IMAGE:
            # Too many blocks to update, reload the entire image
            self.map_scene_view.load_map()
            self.map_scene_view.update_visible_layers(
                self.tabs.currentWidget().visible_layers
            )
        else:
            for y, x in zip(idx_y, idx_x):
                y, x = cast(int, y), cast(int, x)
                self.map_scene_view.blocks.update_block_image_at_padded_position(x, y)

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
        idx_y, idx_x = indices_padded
        if len(idx_y) > self.THRESHOLD_NUM_BLOCKS_TO_RELOAD_IMAGE:
            # Too many blocks to update, reload the entire image
            self.map_scene_view.load_map()
            self.map_scene_view.update_visible_layers(
                self.tabs.currentWidget().visible_layers
            )
        else:
            for y, x in zip(idx_y, idx_x):
                y, x = cast(int, y), cast(int, x)
                self.map_scene_view.levels.update_level_image_at_padded_position(x, y)

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
        assert self.map_scene_view.visible_blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        self.map_scene_view.visible_blocks[
            indices[0] + padded_height, indices[1] + padded_width, layer
        ] = blocks[...]
        indices = np.array(indices).copy()
        indices[0] += padded_height
        indices[1] += padded_width
        match layer:
            case 0:
                self.update_blocks_at_padded_indices(indices)
            case 1:
                self.update_levels_at_padded_indices(indices)
            case _:
                ...  # Invalid layer, do nothing

    def update_map_at(self, x: int, y: int, layers: MapLayers, blocks: Tilemap):
        """Updates the map image with new blocks rooted at a certain position."""
        if not self.main_gui.footer_loaded:
            return
        assert self.map_scene_view.visible_blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.block_images is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()

        self.map_scene_view.visible_blocks[
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
        # if not self.main_gui.footer_loaded:
        #     return
        # smart_shape = self.main_gui.smart_shapes[
        #     self.smart_shapes_tab.current_smart_shape_name
        # ]
        # assert self.main_gui.project is not None
        # template = self.main_gui.project.smart_shape_templates[smart_shape.template]
        # smart_shape.buffer[y : y + blocks.shape[0], x : x + blocks.shape[1]] = blocks
        # padded_width, padded_height = self.main_gui.get_border_padding()
        # map_width, map_height = self.main_gui.get_map_dimensions()
        # # Redraw relevant block pixmaps
        # for (yy, xx), block_idx in np.ndenumerate(blocks[:, :, 0]):
        #     pixmap = template.block_pixmaps[block_idx]
        #     if x + xx in range(map_width) and y + yy in range(map_height):
        #         self.smart_shape_images[
        #             padded_height + y + yy, padded_width + x + xx
        #         ].setPixmap(pixmap)
