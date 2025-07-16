"""Tab for the levels of the map widget."""

from __future__ import annotations

import importlib.resources as resources
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from pymap.gui.map.view import VisibleLayer
from pymap.gui.render import QImage_to_ndarray, split_image_into_tiles
from pymap.gui.types import Tilemap

from ..blocks_like import BlocksLikeTab
from .level_blocks import LevelBlocksScene

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class LevelsTab(BlocksLikeTab):
    """Tab for the levels."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)
        level_layout = QVBoxLayout()
        self.setLayout(level_layout)
        self.level_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_opacity_slider.setMinimum(0)
        self.level_opacity_slider.setMaximum(20)
        self.level_opacity_slider.setSingleStep(1)
        self.level_opacity_slider.setSliderPosition(
            self.map_widget.main_gui.settings.value('map_widget/level_opacity', 30, int)  # type: ignore
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
            str(
                resources.files('pymap.gui.map.tabs.levels').joinpath(
                    'level_blocks.png'
                )
            ),
        )
        self.levels_blocks_rgba = split_image_into_tiles(
            QImage_to_ndarray(self.level_blocks_pixmap.toImage()),
            tile_size=16,
        ).reshape((-1, 16, 16, 4))

        # And split them
        self.level_blocks_pixmaps = [
            self.level_blocks_pixmap.copy((idx % 4) * 16, (idx // 4) * 16, 16, 16)
            for idx in range(0x40)
        ]
        self.level_scene = LevelBlocksScene(self)
        self.level_scene_view = QtWidgets.QGraphicsView()
        self.level_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.level_scene_view.setScene(self.level_scene)
        self.level_scene_view.leaveEvent = (
            lambda event: self.map_widget.info_label.clear()
        )
        level_layout.addWidget(self.level_scene_view)
        item = QGraphicsPixmapItem(
            self.level_blocks_pixmap.scaled(4 * 16 * 2, 16 * 16 * 2)
        )
        item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.level_scene.addItem(item)
        item.setAcceptHoverEvents(False)
        item.hoverLeaveEvent = lambda event: self.map_widget.info_label.setText('')
        self.level_scene.setSceneRect(0, 0, 4 * 16 * 2, 16 * 16 * 2)

    @property
    def connectivity_layer(self) -> int:
        """Returns the connectivity level."""
        return 1

    @property
    def visible_layers(self) -> VisibleLayer:
        """Get the visible layers."""
        return (
            VisibleLayer.BLOCKS
            | VisibleLayer.LEVELS
            | VisibleLayer.BORDER_EFFECT
            | VisibleLayer.GRID
            | VisibleLayer.TRANSPARENT_BACKGROUND
        )

    @property
    def selected_layers(self) -> Tilemap:
        """Returns the selected layers."""
        return np.array([1])

    def change_levels_opacity(self):
        """Changes the opacity of the levels."""
        if not self.map_widget.main_gui.project_loaded:
            return
        assert self.map_widget.main_gui.project is not None, 'Project is not loaded'
        opacity = self.level_opacity_slider.sliderPosition()
        self.map_widget.main_gui.settings.setValue('map_widget/level_opacity', opacity)
        self.map_widget.map_scene_view.levels.update_level_image_opacity()

    def load_project(self) -> None:
        """Loads the project."""
        self.set_selection(np.zeros((1, 1, 2), dtype=int))

    def set_selection(self, selection: Tilemap):
        """Sets the selection.

        Args:
            selection (RGBAImage): The selection.
        """
        selection = selection.copy()
        self.selection = selection
        self.levels_selection_scene.clear()
        if not self.map_widget.header_loaded:
            return
        # Levels selection
        for (y, x), level in np.ndenumerate(selection[:, :, 1]):
            item = QGraphicsPixmapItem(self.level_blocks_pixmaps[level])
            item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
            self.levels_selection_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
        self.levels_selection_scene.setSceneRect(
            0, 0, selection.shape[1] * 16, selection.shape[0] * 16
        )

    def load_map(self):
        """Reloads the map image by using tiles of the map widget."""
