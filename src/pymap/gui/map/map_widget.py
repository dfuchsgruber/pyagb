"""Widget for the map."""

from __future__ import annotations

import importlib.resources as resources
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL.ImageQt import ImageQt
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QHBoxLayout,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pymap.gui import blocks, render
from pymap.gui.blocks import compute_blocks
from pymap.gui.types import MapLayers

from .auto_shape import AutoScene
from .blocks import BlocksScene
from .border import BorderScene
from .level_blocks import LevelBlocksScene
from .map import MapScene

if TYPE_CHECKING:
    from ..main.gui import PymapGui


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
        self.blocks = None
        self.selection = None
        self.layers = np.array(0)
        self.undo_stack = QtGui.QUndoStack()

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
        splitter.addWidget(self.map_scene_view)

        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label, 2, 1, 1, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)

        # Divide into a widget for blocks and levels
        self.tabs = QTabWidget()

        self.tabs.addTab(self._setup_blocks_widget(), 'Blocks')
        self.tabs.addTab(self._setup_levels_widget(), 'Level')
        self.tabs.currentChanged.connect(self.tab_changed)

        splitter.addWidget(self.tabs)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        splitter.restoreState(
            self.main_gui.settings.value(
                'map_widget/horizontal_splitter_state', bytes(), type=bytes
            )  # type: ignore
        )
        splitter.splitterMoved.connect(
            lambda: self.main_gui.settings.setValue(
                'map_widget/horizontal_splitter_state', splitter.saveState()
            )
        )
        self.load_header()  # Disables all widgets

    def _setup_blocks_widget(self) -> QWidget:
        """Sets up the blocks widget to the right."""
        blocks_widget = QSplitter(Qt.Orientation.Vertical)

        splitter_selection_and_border = QSplitter(Qt.Orientation.Horizontal)
        splitter_selection_and_border.restoreState(
            self.main_gui.settings.value(
                'map_widget/selection_and_border_splitter_state', bytes(), type=bytes
            )  # type: ignore
        )
        splitter_selection_and_border.splitterMoved.connect(
            lambda: self.main_gui.settings.setValue(
                'map_widget/selection_and_border_splitter_state',
                splitter_selection_and_border.saveState(),
            )
        )

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
        self.show_border.toggled.connect(self.load_map)
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
        self.select_levels.toggled.connect(self.update_layers)
        group_selection_layout.addWidget(self.select_levels, 3, 1, 1, 1)

        splitter_selection_and_border.addWidget(group_selection)
        splitter_selection_and_border.addWidget(group_border)

        self.auto_group = QtWidgets.QGroupBox('Smart Shapes')
        auto_group_layout = QHBoxLayout()
        self.auto_group.setLayout(auto_group_layout)
        self.auto_shapes_scene = AutoScene(self)
        self.auto_shapes_scene_view = QtWidgets.QGraphicsView()
        self.auto_shapes_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.auto_shapes_scene_view.setScene(self.auto_shapes_scene)
        self.auto_shapes_scene_view.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        auto_group_layout.addWidget(self.auto_shapes_scene_view)
        # selection_and_auto_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        blocks_widget.addWidget(self.auto_group)

        group_blocks = QWidget()
        group_blocks_layout = QtWidgets.QGridLayout()
        group_blocks.setLayout(group_blocks_layout)
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QtWidgets.QGraphicsView()
        self.blocks_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        group_blocks_layout.addWidget(self.blocks_scene_view)
        group_blocks.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        blocks_widget.addWidget(group_blocks)

        blocks_widget.setStretchFactor(0, 0)  # Border
        blocks_widget.setStretchFactor(1, 0)  # Selection and Auto
        blocks_widget.setStretchFactor(2, 1)  # Blocks

        blocks_widget.restoreState(
            self.main_gui.settings.value(
                'map_widget/blocks_splitter_state', bytes(), type=bytes
            )  # type: ignore
        )
        blocks_widget.splitterMoved.connect(
            lambda: self.main_gui.settings.setValue(
                'map_widget/blocks_splitter_state', blocks_widget.saveState()
            )
        )

        return blocks_widget

    def _setup_levels_widget(self) -> QWidget:
        """Sets up the levels widget to the right."""
        level_widget = QWidget()
        level_layout = QVBoxLayout()
        level_widget.setLayout(level_layout)
        self.level_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_opacity_slider.setMinimum(0)
        self.level_opacity_slider.setMaximum(20)
        self.level_opacity_slider.setSingleStep(1)
        self.level_opacity_slider.setSliderPosition(
            self.main_gui.settings.value('map_widget/level_opacity', 30, int)  # type: ignore
        )
        self.level_opacity_slider.valueChanged.connect(self.change_levels_opacity)
        level_opacity_group = QtWidgets.QGroupBox('Opacity')
        level_opactiy_group_layout = QVBoxLayout()
        level_opacity_group.setLayout(level_opactiy_group_layout)
        level_opactiy_group_layout.addWidget(self.level_opacity_slider)
        level_layout.addWidget(level_opacity_group)

        group_selection = QtWidgets.QGroupBox('Selection')
        group_selection_layout = QtWidgets.QGridLayout()
        group_selection.setLayout(group_selection_layout)
        self.levels_selection_scene = QGraphicsScene()
        self.levels_selection_scene_view = QtWidgets.QGraphicsView()
        self.levels_selection_scene_view.setScene(self.levels_selection_scene)
        self.levels_selection_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        group_selection_layout.addWidget(self.levels_selection_scene_view, 1, 1, 2, 1)
        level_layout.addWidget(group_selection)

        # Load level gfx
        self.level_blocks_pixmap = QPixmap(
            str(resources.files('pymap.gui.map').joinpath('level_blocks.png')),
        )
        # And split them
        self.level_blocks_pixmaps = [
            self.level_blocks_pixmap.copy((idx % 4) * 16, (idx // 4) * 16, 16, 16)
            for idx in range(0x40)
        ]
        self.level_scene = LevelBlocksScene(self)
        self.level_scene_view = QtWidgets.QGraphicsView()
        self.level_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.level_scene_view.setScene(self.level_scene)
        level_layout.addWidget(self.level_scene_view)
        item = QGraphicsPixmapItem(
            self.level_blocks_pixmap.scaled(4 * 16 * 2, 16 * 16 * 2)
        )
        self.level_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        item.hoverLeaveEvent = lambda event: self.info_label.setText('')
        self.level_scene.setSceneRect(0, 0, 4 * 16 * 2, 16 * 16 * 2)
        return level_widget

    @property
    def header_loaded(self) -> bool:
        """Whether the header is loaded.

        Returns:
            bool: Whether the header is loaded.
        """
        return self.main_gui.header_loaded

    def tab_changed(self):
        """Triggered when the user switches from the blocks to levels tab."""
        self.update_layers()
        self.load_map()

    def update_layers(self):
        """Updates the current layers."""
        if self.tabs.currentIndex() == 0:
            if self.select_levels.isChecked():
                self.layers = np.array([0, 1])
            else:
                self.layers = np.array([0])
        else:
            self.layers = np.array([1])

    def change_levels_opacity(self):
        """Changes the opacity of the levels."""
        if not self.main_gui.project_loaded:
            return
        assert self.main_gui.project is not None, 'Project is not loaded'
        opacity = self.level_opacity_slider.sliderPosition()
        self.main_gui.settings.setValue('map_widget/level_opacity', opacity)
        self.load_map()

    def load_project(self, *args: Any):
        """Update project related widgets."""
        if not self.main_gui.project_loaded:
            return
        assert self.main_gui.project is not None, 'Project is not loaded'
        self.set_blocks_selection(np.zeros((1, 1, 2), dtype=int))
        self.set_levels_selection(np.zeros((1, 1, 2), dtype=int))
        self.load_header()

    def load_header(self, *args: Any):
        """Updates the entire header related widgets."""
        # Clear graphics
        self.load_map()
        self.load_border()
        self.load_blocks()
        self.load_auto_shapes()

        if self.main_gui.project is None or self.main_gui.header is None:
            # Reset all widgets
            self.blocks = None
            self.show_border.setEnabled(False)
            self.select_levels.setEnabled(False)
        else:
            # Update selection blocks
            assert self.selection is not None, 'Selection is not loaded'
            self.set_selection(self.selection)
            self.show_border.setEnabled(True)
            self.select_levels.setEnabled(True)

    def update_grid(self):
        """Updates the grid."""
        self.map_scene.update_grid()

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

        # Create a pixel map for each block
        self.map_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        self.level_images = np.empty_like(self.blocks[:, :, 1], dtype=object)
        for (y, x), block_idx in np.ndenumerate(self.blocks[:, :, 0]):
            # Draw the blocks
            assert self.main_gui.blocks is not None, 'Blocks are not loaded'
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
            self.map_images[y, x] = item
        for (y, x), level in np.ndenumerate(self.blocks[:, :, 1]):
            if x in range(padded_width, padded_width + map_width) and y in range(
                padded_height, padded_height + map_height
            ):
                # Draw the pixmaps
                pixmap = self.level_blocks_pixmaps[level]
                item = QGraphicsPixmapItem(pixmap)
                item.setAcceptHoverEvents(True)
                opacity = QGraphicsOpacityEffect()
                opacity.setOpacity(self.level_opacity_slider.sliderPosition() / 20)
                item.setGraphicsEffect(opacity)
                if self.tabs.currentIndex() == 1:
                    self.map_scene.addItem(item)
                    item.setPos(16 * x, 16 * y)
                self.level_images[y, x] = item

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
        self.update_grid()

    def update_map(self, x: int, y: int, layers: MapLayers, blocks: NDArray[np.int_]):
        """Updates the map image with new blocks rooted at a certain position."""
        if not self.main_gui.footer_loaded:
            return
        assert self.blocks is not None, 'Blocks are not loaded'
        assert self.main_gui.blocks is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        self.blocks[
            padded_height + y : padded_height + y + blocks.shape[0],
            padded_width + x : padded_width + x + blocks.shape[1],
            layers,
        ] = blocks[:, :, layers]
        assert isinstance(layers, Iterable), 'Layers is not an iterable'
        # Redraw relevant pixel maps
        if 0 in layers:
            for (yy, xx), block_idx in np.ndenumerate(blocks[:, :, 0]):
                pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
                self.map_images[
                    padded_height + y + yy, padded_width + x + xx
                ].setPixmap(pixmap)
        if 1 in layers:
            for (yy, xx), level in np.ndenumerate(blocks[:, :, 1]):
                if x + xx in range(map_width) and y + yy in range(map_height):
                    self.level_images[
                        padded_height + y + yy, padded_width + x + xx
                    ].setPixmap(self.level_blocks_pixmaps[level])

    def load_blocks(self):
        """Loads the block pool."""
        self.blocks_scene.clear()
        if not self.main_gui.footer_loaded:
            return
        assert self.main_gui.blocks is not None, 'Blocks are not loaded'
        self.blocks_image = QPixmap.fromImage(
            ImageQt(render.draw_blocks(self.main_gui.blocks))
        )
        item = QGraphicsPixmapItem(self.blocks_image)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        item.hoverLeaveEvent = lambda event: self.info_label.setText('')

    def load_border(self):
        """Loads the border."""
        self.border_scene.clear()
        if not self.main_gui.footer_loaded:
            return
        assert self.main_gui.blocks is not None, 'Blocks are not loaded'
        border_blocks = self.main_gui.get_borders()
        self.border_image = QPixmap.fromImage(
            ImageQt(render.draw_blocks(self.main_gui.blocks, border_blocks))
        )
        self.border_scene.addPixmap(self.border_image)
        self.border_scene.setSceneRect(
            0, 0, border_blocks.shape[1] * 16, border_blocks.shape[0] * 16
        )

    def load_auto_shapes(self):
        """Loads automatic shapes."""
        if not self.main_gui.project_loaded:
            return
        self.auto_shapes_scene.update_pixmap()

    def set_selection(self, selection: NDArray[np.int_]):
        """Sets currently selected blocks or level blocks depending on the tab."""
        if self.tabs.currentIndex() == 0:
            self.set_blocks_selection(selection)
        elif self.tabs.currentIndex() == 1:
            self.set_levels_selection(selection)
        else:
            raise RuntimeError(f'Unsupported tab {self.tabs.currentIndex() }')

    def set_blocks_selection(self, selection: NDArray[np.int_]):
        """Sets currently selected blocks."""
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        # Block selection
        assert self.main_gui.blocks is not None, 'Blocks are not loaded'
        selection_pixmap = QPixmap.fromImage(
            ImageQt(render.draw_blocks(self.main_gui.blocks, self.selection))
        )
        item = QGraphicsPixmapItem(selection_pixmap)
        self.selection_scene.addItem(item)
        self.selection_scene.setSceneRect(
            0, 0, selection_pixmap.width(), selection_pixmap.height()
        )

    def set_levels_selection(self, selection: NDArray[np.int_]):
        """Sets currently selected level blocks."""
        selection = selection.copy()
        self.levels_selection = selection
        self.levels_selection_scene.clear()
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.selection is None
        ):
            return
        # Levels selection
        for (y, x), level in np.ndenumerate(selection[:, :, 1]):
            item = QGraphicsPixmapItem(self.level_blocks_pixmaps[level])
            self.levels_selection_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
        self.levels_selection_scene.setSceneRect(
            0, 0, selection.shape[1] * 16, selection.shape[0] * 16
        )
