"""Widget that shows the map."""

from __future__ import annotations

from enum import IntFlag, auto, unique
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QWidget,
)

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType
from pymap.gui import blocks, properties
from pymap.gui.blocks import compute_blocks
from pymap.gui.event import EventToImage, NullEventToImage
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
        self.events_group = None
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

    def _load_event_to_image_backend(self):
        """Loads the event to image backend."""
        project = self.main_gui.project
        event_to_image: None | EventToImage = None
        if project is not None:
            backend = project.config['pymap']['display']['event_to_image_backend']

            if backend is not None:
                with project.project_dir():
                    with open(Path(backend)) as f:
                        namespace: dict[str, object] = {}
                        exec(f.read(), namespace)
                        get_event_to_image = namespace['get_event_to_image']
                        assert isinstance(get_event_to_image, Callable)
                        event_to_image = cast(EventToImage, get_event_to_image())
        if event_to_image is None:
            self.event_to_image = NullEventToImage()
        else:
            self.event_to_image = event_to_image

    def load_project(self):
        """Loads the project."""
        self._load_event_to_image_backend()

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
        self.add_event_images()

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

    @property
    def show_event_images(self) -> bool:
        """Returns whether the event images are shown."""
        return cast(
            bool, self.main_gui.settings.value('event_widget/show_pictures', True, bool)
        )

    def add_event_images(self):
        """Adds all event images to the scene."""
        assert self.main_gui.project is not None
        from pymap.gui.map.tabs.events.event_to_image import EventImage

        self.events_group = QGraphicsItemGroup()
        self.event_images: dict[str, list[QGraphicsItem]] = {}
        for event_type in self.main_gui.project.config['pymap']['header']['events']:
            self.event_images[event_type['name']] = []
            events = self.main_gui.get_events(event_type)
            for event in events:
                event_image = self.event_to_image.event_to_image(
                    event, event_type, self.main_gui.project
                )
                padded_x, padded_y = self.main_gui.get_border_padding()
                x, y = self.pad_coordinates(
                    properties.get_member_by_path(event, event_type['x_path']),
                    properties.get_member_by_path(event, event_type['y_path']),
                    padded_x,
                    padded_y,
                )
                if event_image is not None and self.show_event_images:
                    event_image = EventImage(*event_image)
                    pixmap = QPixmap.fromImage(ndarray_to_QImage(event_image.image))
                    item = QGraphicsPixmapItem(pixmap)
                    item.setPos(
                        16 * (x) + event_image.x_offset, 16 * (y) + event_image.y_offset
                    )
                else:
                    item = self._get_event_image_rectangle(event_type)
                    item.setPos(16 * x, 16 * y)
                item.setAcceptHoverEvents(True)
                self.events_group.addToGroup(item)
                self.event_images[event_type['name']].append(item)
        self.addItem(self.events_group)

    @staticmethod
    def _get_event_image_rectangle(
        event_type: PymapEventConfigType,
    ) -> QGraphicsItemGroup:
        """Creates a rectangle for the event image.

        Args:
            event_type (PymapEventConfigType): The event type.

        Returns:
            QGraphicsItemGroup: The group.
        """
        group = QGraphicsItemGroup()
        color = QColor.fromRgbF(*(event_type['box_color']))
        rect = QGraphicsRectItem(0, 0, 16, 16)
        rect.setBrush(QBrush(color))
        rect.setPen(QPen(0))
        text = QGraphicsTextItem(event_type['name'][0])
        text.setPos(
            6 - text.sceneBoundingRect().width() / 2,
            6 - text.sceneBoundingRect().height() / 2,
        )

        font = QFont('Ubuntu')
        font.setBold(True)
        font.setPixelSize(16)
        text.setFont(font)
        text.setDefaultTextColor(QColor.fromRgbF(*(event_type['text_color'])))
        group.addToGroup(rect)
        group.addToGroup(text)
        return group

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
        if self.events_group is not None:
            self.events_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.EVENTS) > 0
            )

    @staticmethod
    def pad_coordinates(
        x: ModelValue, y: ModelValue, padded_x: int, padded_y: int
    ) -> tuple[int, int]:
        """Tries to transform the text string of an event to integer coordinates."""
        x, y = str(x), str(y)  # This enables arbitrary bases
        try:
            x = int(x, 0)
            y = int(y, 0)
        except ValueError:
            return (
                -10000,
                -10000,
            )  # This is hacky but prevents the events from being rendered
        return (x + padded_x), (y + padded_y)
