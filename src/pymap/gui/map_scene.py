"""Widget that shows the map."""

from __future__ import annotations

from enum import IntFlag, auto, unique
from typing import TYPE_CHECKING

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QWidget,
)
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap

from pymap.gui import blocks
from pymap.gui.blocks import compute_blocks
from pymap.gui.render import ndarray_to_QImage
from pymap.gui.types import Tilemap

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

import numpy as np


class MapScene(QGraphicsScene):
    """Scene that will show the map."""

    @unique
    class VisibleLayer(IntFlag):
        """Layers that can be visible."""

        BLOCKS = auto()
        LEVELS = auto()
        BORDER_EFFECT = auto()
        SMART_SHAPE = auto()
        EVENTS = auto()
        CONNECTIONS = auto()

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the map scene.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.grid_group = None
        self.blocks_group = None
        self.levels_group = None
        self.border_effect_group = None
        # Note that self.blocks is a padded tilemap of what is visible
        # and does not correspond to the actual blocks of the footer
        # which are still held in the main_gui.footer
        self.blocks: Tilemap | None = None

    def clear(self) -> None:
        """Clears the scene."""
        super().clear()
        self.grid_group = None
        self.blocks_group = None
        self.levels_group = None
        self.border_effect_group = None

    def update_grid(self):
        """Updates the grid of the scene."""
        if self.grid_group is not None:
            self.removeItem(self.grid_group)
        if self.main_gui.grid_visible:
            self.grid_group = QGraphicsItemGroup()
            for x in range(0, int(self.width()), 16):
                self.grid_group.addToGroup(self.addLine(x, 0, x, self.height()))
            for y in range(0, int(self.height()), 16):
                self.grid_group.addToGroup(self.addLine(0, y, self.width(), y))
            self.addItem(self.grid_group)
        else:
            self.grid_group = None

    def load_map(self):
        """(Re-)Loads the entire map scene.

        First, it computes the visible blocks and then adds computes all layers
        and adds them to the scene.
        """
        self.clear()
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.compute_blocks()
        self.add_block_images()
        self.add_level_images()
        self.add_border_effect()

    def compute_blocks(self):
        """Computes the visible block tilemap."""
        assert self.main_gui.project is not None
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Crop the visible blocks from all blocks including the border
        self.blocks = compute_blocks(self.main_gui.footer, self.main_gui.project)  #
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

    def add_block_images(self):
        """Adds all block images to the scene."""
        assert self.blocks is not None, 'Blocks are not loaded'
        self.block_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        self.blocks_group = QGraphicsItemGroup()
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
            self.blocks_group.addToGroup(item)
        self.addItem(self.blocks_group)

    def update_block_image_at_padded_position(self, x: int, y: int):
        """Updates the block image at the given padded position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        assert self.blocks is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        if x in range(padded_width, padded_width + map_width) and y in range(
            padded_height, padded_height + map_height
        ):
            # Draw the blocks
            block_idx: int = self.blocks[y, x, 0]
            assert self.main_gui.block_images is not None, 'Blocks are not loaded'
            pixmap = QPixmap.fromImage(
                ndarray_to_QImage(self.main_gui.block_images[block_idx])
            )
            item: QGraphicsPixmapItem = self.block_images[y, x]
            item.setPixmap(pixmap)

    def add_level_images(self):
        """Adds all level images to the scene."""
        assert self.blocks is not None, 'Blocks are not loaded'
        self.level_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        self.levels_group = QGraphicsItemGroup()
        for (y, x), level in np.ndenumerate(self.blocks[:, :, 1]):
            if x in range(padded_width, padded_width + map_width) and y in range(
                padded_height, padded_height + map_height
            ):
                # Draw the pixmaps
                pixmap = self.main_gui.map_widget.levels_tab.level_blocks_pixmaps[level]
                item = QGraphicsPixmapItem(pixmap)
                item.setAcceptHoverEvents(True)
                item.setPos(16 * x, 16 * y)
                self.level_images[y, x] = item
                self.levels_group.addToGroup(item)

        self.level_image_opacity_effect = QGraphicsOpacityEffect()
        self.levels_group.setGraphicsEffect(self.level_image_opacity_effect)
        self.update_level_image_opacity()
        self.addItem(self.levels_group)

    def update_level_image_opacity(self):
        """Updates the opacity of the level images."""
        self.level_image_opacity_effect.setOpacity(
            self.main_gui.map_widget.levels_tab.level_opacity_slider.sliderPosition()
            / 20
        )

    def update_level_image_at_padded_position(self, x: int, y: int):
        """Updates the level image at the given padded position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        assert self.blocks is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        if x in range(padded_width, padded_width + map_width) and y in range(
            padded_height, padded_height + map_height
        ):
            # Draw the pixmaps
            level: int = self.blocks[y, x, 1]
            pixmap = self.main_gui.map_widget.levels_tab.level_blocks_pixmaps[level]
            item: QGraphicsPixmapItem = self.level_images[y, x]
            item.setPixmap(pixmap)

    def add_border_effect(self):
        """Adds the opacity effect for borders."""
        assert self.main_gui.project is not None, 'Project is not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Apply shading to border parts by adding opaque rectangles
        border_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['border_color'])
        )
        self.border_effect_group = QGraphicsItemGroup()
        self.north_border = self.addRect(
            0,
            0,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.south_border = self.addRect(
            0,
            (padded_height + map_height) * 16,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.west_border = self.addRect(
            0,
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.east_border = self.addRect(
            16 * (padded_width + map_width),
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.border_effect_group.addToGroup(self.north_border)
        self.border_effect_group.addToGroup(self.south_border)
        self.border_effect_group.addToGroup(self.west_border)
        self.border_effect_group.addToGroup(self.east_border)
        self.addItem(self.border_effect_group)

    def update_scene_rect(self):
        """Updates the scene rectangle bounds."""
        assert self.main_gui.project is not None, 'Project is not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        self.setSceneRect(
            0,
            0,
            16 * (2 * padded_width + map_width),
            16 * (2 * padded_height + map_height),
        )

    def update_visible_layers(self, visible_layers: VisibleLayer):
        """Shows / Hides certain layers.

        Args:
            visible_layers (Layer): Mask for visible layers.
        """
        if self.blocks_group is not None:
            self.blocks_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.BLOCKS) > 0
            )
        if self.border_effect_group is not None:
            self.border_effect_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.BORDER_EFFECT) > 0
            )
        if self.levels_group is not None:
            self.levels_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.LEVELS) > 0
            )
