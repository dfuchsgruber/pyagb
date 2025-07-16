"""Widget that shows the map."""

from __future__ import annotations

from enum import IntFlag, auto, unique
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
    QWidget,
)

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType
from pymap.gui import blocks, properties
from pymap.gui.blocks import (
    compute_blocks,
    connection_get_blocks,
    connection_get_connection_type,
    connection_get_offset,
)
from pymap.gui.map.tabs.events.event_image import EventImage
from pymap.gui.render import ndarray_to_QImage
from pymap.gui.transparent import QGraphicsSceneWithTransparentBackground
from pymap.gui.types import ConnectionType, Tilemap, visible_connection_directions

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

import numpy as np


class MapScene(QGraphicsSceneWithTransparentBackground):
    """Scene that will show the map."""

    @unique
    class VisibleLayer(IntFlag):
        """Layers that can be visible."""

        BLOCKS = auto()
        LEVELS = auto()
        BORDER_EFFECT = auto()
        SMART_SHAPE = auto()
        EVENTS = auto()
        SELECTED_EVENT = auto()
        CONNECTIONS = auto()
        CONNECTION_RECTANGLES = auto()

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the map scene.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.setItemIndexMethod(self.ItemIndexMethod.NoIndex)
        self.main_gui = main_gui
        self.visible_layers = self.VisibleLayer(0)
        self.transparent_background_group = None
        self.grid_group = None
        self.blocks_group = None
        self.levels_group = None
        self.border_effect_group = None
        self.events_group = None
        self.selected_event_group = None
        self.connection_rectangles_group = None
        # Note that self.blocks is a padded tilemap of what is visible
        # and does not correspond to the actual blocks of the footer
        # which are still held in the main_gui.footer
        self.blocks: Tilemap | None = None

    def clear(self) -> None:
        """Clears the scene."""
        super().clear()
        self.transparent_background_group = None
        self.grid_group = None
        self.blocks_group = None
        self.levels_group = None
        self.border_effect_group = None
        self.events_group = None
        self.selected_event_group = None
        self.connection_rectangles_group = None

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
            self._group_set_flags(self.grid_group)
        else:
            self.grid_group = None

    def load_project(self):
        """Loads the project."""
        ...

    def load_map(self):
        """(Re-)Loads the entire map scene.

        First, it computes the visible blocks and then adds computes all layers
        and adds them to the scene.
        """
        self.clear()
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.compute_blocks()
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        self.add_transparent_background(
            16 * (map_width + 2 * padded_width), 16 * (map_height + 2 * padded_height)
        )
        self.add_block_images()
        self.add_level_images()
        self.add_border_effect()
        self.add_event_images()
        self.add_selected_event_images()
        self.add_connection_rectangles()

    def compute_blocks(self):
        """Computes the visible block tilemap."""
        assert self.main_gui.project is not None
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Crop the visible blocks from all blocks including the border
        self.blocks = compute_blocks(self.main_gui.footer, self.main_gui.project)  #
        connections = self.main_gui.get_connections()
        for connection in connections:
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
            item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
            item.setAcceptHoverEvents(False)
            item.setPos(16 * x, 16 * y)
            self.block_images[y, x] = item
            self.blocks_group.addToGroup(item)
        self.addItem(self.blocks_group)
        self._group_set_flags(self.blocks_group)

    def _group_set_flags(self, group: QGraphicsItemGroup):
        group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        group.setAcceptHoverEvents(False)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, False)
        group.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    def update_block_image_at_padded_position(self, x: int, y: int):
        """Updates the block image at the given padded position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        assert self.blocks is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        if x in range(2 * padded_width + map_width) and y in range(
            2 * padded_height + map_height
        ):
            # Draw the blocks
            block_idx: int = self.blocks[y, x, 0]
            assert self.main_gui.block_images is not None, 'Blocks are not loaded'
            pixmap = QPixmap.fromImage(
                ndarray_to_QImage(self.main_gui.block_images[block_idx])
            )
            item: QGraphicsPixmapItem = self.block_images[y, x]
            if item.pixmap().cacheKey() != pixmap.cacheKey():
                item.setPixmap(pixmap)

    def add_level_images(self):
        """Adds all level images to the scene."""
        assert self.blocks is not None, 'Blocks are not loaded'
        self.level_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        self.levels_group = QGraphicsItemGroup()
        for (y, x), level in np.ndenumerate(self.blocks[:, :, 1]):
            # Draw the pixmaps
            pixmap = self.main_gui.map_widget.levels_tab.level_blocks_pixmaps[level]
            item = QGraphicsPixmapItem(pixmap)
            item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
            item.setAcceptHoverEvents(False)
            item.setPos(16 * x, 16 * y)
            self.level_images[y, x] = item
            self.levels_group.addToGroup(item)

        self.level_image_opacity_effect = QGraphicsOpacityEffect()
        self.levels_group.setGraphicsEffect(self.level_image_opacity_effect)
        self.update_level_image_opacity()
        self.addItem(self.levels_group)
        self._group_set_flags(self.levels_group)

    @property
    def visible_connection_idxs(self) -> dict[ConnectionType, int]:
        """Returns the visible connection indexes."""
        assert self.main_gui.project is not None
        visible_connections = {
            connection_type: next(
                (
                    idx
                    for idx, connection in enumerate(self.main_gui.get_connections())
                    if connection is not None
                    and str(
                        connection_get_connection_type(
                            connection, self.main_gui.project
                        )
                    )
                    == str(connection_type)
                ),
                None,
            )
            for connection_type in ConnectionType
        }
        return {
            connection_type: idx
            for connection_type, idx in visible_connections.items()
            if idx is not None
        }

    def connection_rectangle_get_position_and_dimensions(
        self, connection_idx: int
    ) -> tuple[int, int, int, int]:
        """Returns the position and dimensions of a connection.

        Args:
            connection_idx (int): The index of the connection.

        Returns:
            tuple[int, int, int, int]: The position and dimensions of the connection.
        """
        assert self.main_gui.project is not None
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        connections = self.main_gui.get_connections()
        connection = connections[connection_idx]
        if connection is None:
            return 0, 0, 0, 0
        connection_blocks = connection_get_blocks(
            connection,
            self.main_gui.project,
        )
        connection_type = connection_get_connection_type(
            connection,
            self.main_gui.project,
        )
        connection_offset = connection_get_offset(
            connection,
            self.main_gui.project,
        )
        if connection_blocks is None or connection_offset is None:
            return 0, 0, 0, 0
        connection_width, connection_height = (
            connection_blocks.shape[1],
            connection_blocks.shape[0],
        )

        match connection_type:
            case ConnectionType.NORTH:
                rect = (
                    16 * (padded_width + connection_offset),
                    16 * (padded_height - connection_height),
                    16 * connection_width,
                    16 * connection_height,
                )
            case ConnectionType.SOUTH:
                rect = (
                    16 * (padded_width + connection_offset),
                    16 * (padded_height + map_height),
                    16 * connection_width,
                    16 * connection_height,
                )
            case ConnectionType.EAST:
                rect = (
                    16 * (padded_width + map_width),
                    16 * (padded_height + connection_offset),
                    16 * connection_width,
                    16 * connection_height,
                )
            case ConnectionType.WEST:
                rect = (
                    16 * (padded_width - connection_width),
                    16 * (padded_height + connection_offset),
                    16 * connection_width,
                    16 * connection_height,
                )
            case _:
                raise ValueError(f'Invalid connection type {connection_type}')

        return self.fix_rectangle(
            *rect,
            16 * (map_width + 2 * padded_width),
            16 * (map_height + 2 * padded_height),
        )

    def add_connection_rectangles(self):
        """Adds the rectangles around connections to the scene."""
        assert self.main_gui.project is not None and self.main_gui.header is not None
        self.connection_rectangles_group = QGraphicsItemGroup()
        connection_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['connection_color'])
        )
        connection_border_color = QColor.fromRgbF(
            *(
                self.main_gui.project.config['pymap']['display'][
                    'connection_border_color'
                ]
            )
        )
        self._connection_rectangles: dict[ConnectionType, QGraphicsRectItem] = {}
        for connection_type in visible_connection_directions:
            if connection_type not in self.visible_connection_idxs:
                rect = 0, 0, 0, 0
            else:
                rect = self.connection_rectangle_get_position_and_dimensions(
                    self.visible_connection_idxs[connection_type]
                )
            self._connection_rectangles[connection_type] = self.addRect(
                *rect,
                pen=QPen(connection_border_color),
                brush=QBrush(connection_color),
            )
            self.connection_rectangles_group.addToGroup(
                self._connection_rectangles[connection_type]
            )
        self.addItem(self.connection_rectangles_group)
        self._group_set_flags(self.connection_rectangles_group)

    def update_connection_rectangles(self):
        """Updates the connection rectangles."""
        assert self.main_gui.project is not None
        if self.connection_rectangles_group is None:
            return
        for (
            connection_type,
            rectangle_graphics_item,
        ) in self._connection_rectangles.items():
            if connection_type not in self.visible_connection_idxs:
                rect = 0, 0, 0, 0
            else:
                rect = self.connection_rectangle_get_position_and_dimensions(
                    self.visible_connection_idxs[connection_type]
                )
            rectangle_graphics_item.setRect(*rect)

    def update_selected_connection(self, connection_idx: int):
        """Updates the selected connection rectangle.

        Args:
            connection_idx (int | None): The index of the connection.
        """
        assert self.main_gui.project is not None
        if self.connection_rectangles_group is None or connection_idx < 0:
            return
        connection_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['connection_color'])
        )
        connection_active_color = QColor.fromRgbF(
            *(
                self.main_gui.project.config['pymap']['display'][
                    'connection_active_color'
                ]
            )
        )
        connection_border_color = QColor.fromRgbF(
            *(
                self.main_gui.project.config['pymap']['display'][
                    'connection_border_color'
                ]
            )
        )
        connection_active_border_color = QColor.fromRgbF(
            *(
                self.main_gui.project.config['pymap']['display'][
                    'connection_active_border_color'
                ]
            )
        )
        idx_to_visible_connection_type = {
            idx: connection_type
            for connection_type, idx in self.visible_connection_idxs.items()
        }
        selected_connection_type = idx_to_visible_connection_type.get(
            connection_idx, None
        )
        for connection_type, rect in self._connection_rectangles.items():
            if connection_type == selected_connection_type:
                rect.setPen(QPen(connection_active_border_color, 2.0))
                rect.setBrush(QBrush(connection_active_color))
            else:
                rect.setPen(QPen(connection_border_color))
                rect.setBrush(QBrush(connection_color))

    @property
    def show_event_images(self) -> bool:
        """Returns whether the event images are shown."""
        return cast(
            bool, self.main_gui.settings.value('event_widget/show_pictures', True, bool)
        )

    def _event_to_qgraphics_item(
        self, event: ModelValue, event_type: PymapEventConfigType
    ) -> QGraphicsItem:
        """Converts an event to a QGraphicsItem.

        Args:
            event (ModelValue): The event.
            event_type (PymapEventConfigType): The event type.

        Returns:
            QGraphicsItem | None: The QGraphicsItem or None if no image is available.
        """
        assert self.main_gui.project is not None
        event_image = self.main_gui.project.backend.event_to_image(
            event,
            event_type,
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
            item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
            item.setPos(
                16 * (x) + event_image.x_offset, 16 * (y) + event_image.y_offset
            )
        else:
            item = self._get_event_image_rectangle(event_type)
            item.setPos(16 * x, 16 * y)
        return item

    def update_event_image(self, event_type: PymapEventConfigType, event_idx: int):
        """Updates a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
        """
        assert self.main_gui.project is not None
        if self.events_group is None:
            return
        item_old = self.event_images[event_type['datatype']][event_idx]
        if item_old in self.events_group.childItems():
            # Remove the item from the group and the scene
            self.events_group.removeFromGroup(item_old)
            self.removeItem(item_old)
            del item_old  # should not be necessary, but just in case

        event = self.main_gui.get_event(event_type, event_idx)
        item = self._event_to_qgraphics_item(event, event_type)
        item.setAcceptHoverEvents(False)
        self.events_group.addToGroup(item)
        self.event_images[event_type['datatype']][event_idx] = item

    def remove_event_image(self, event_type: PymapEventConfigType, event_idx: int):
        """Removes a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
        """
        assert self.main_gui.project is not None
        if self.events_group is None:
            return
        item = self.event_images[event_type['datatype']][event_idx]
        if item in self.events_group.childItems():
            # Remove the item from the group and the scene
            self.events_group.removeFromGroup(item)
            self.removeItem(item)
            del item
        # Remove the item from the list
        self.event_images[event_type['datatype']].pop(event_idx)

    def insert_event_image(
        self, event_type: PymapEventConfigType, event_idx: int, event: ModelValue
    ):
        """Inserts a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
            event (ModelValue): The event.
        """
        assert self.main_gui.project is not None
        if self.events_group is None:
            return
        item = self._event_to_qgraphics_item(event, event_type)
        item.setAcceptHoverEvents(False)
        self.events_group.addToGroup(item)
        self.event_images[event_type['datatype']].insert(event_idx, item)

    def add_event_images(self):
        """Adds all event images to the scene."""
        assert self.main_gui.project is not None
        self.events_group = QGraphicsItemGroup()
        self.event_images: dict[str, list[QGraphicsItem]] = {}
        for event_type in self.main_gui.project.config['pymap']['header'][
            'events'
        ].values():
            self.event_images[event_type['datatype']] = []
            events = self.main_gui.get_events(event_type)
            for event in events:
                item = self._event_to_qgraphics_item(event, event_type)
                item.setAcceptHoverEvents(False)
                self.events_group.addToGroup(item)
                self.event_images[event_type['datatype']].append(item)
        self.addItem(self.events_group)
        self.events_group.setVisible(
            (self.visible_layers & MapScene.VisibleLayer.EVENTS) > 0
        )
        self.events_group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def update_event_images(self):
        """Updates all event images by recomputing them all."""
        if self.events_group is None:
            return
        self.removeItem(self.events_group)
        self.add_event_images()

    def add_selected_event_images(self):
        """Adds the selected event images to the scene."""
        assert self.main_gui.project is not None
        self.selected_event_group = QGraphicsItemGroup()
        rect = QGraphicsRectItem(0, 0, 16, 16)
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        rect.setPen(QPen(color, 2.0))
        rect.setBrush(Qt.BrushStyle.NoBrush)
        self.selected_event_group.addToGroup(rect)
        self.addItem(self.selected_event_group)
        self._group_set_flags(self.selected_event_group)

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
        text = QGraphicsTextItem(event_type['display_letter'][0])
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

    def update_selected_event_image(
        self, event_type: PymapEventConfigType, event_idx: int | None
    ):
        """Updates the selected event image (the red rectangle).

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index. If None or -1, the event is hidden.
        """
        assert self.main_gui.project is not None
        if self.selected_event_group is None:
            return
        if event_idx is None or event_idx < 0:
            # Hide the selected event group
            self.selected_event_group.setVisible(False)
            return
        # Show the selected event group
        if self.visible_layers & MapScene.VisibleLayer.SELECTED_EVENT:
            self.selected_event_group.setVisible(True)

        event = self.main_gui.get_event(event_type, event_idx)
        padded_x, padded_y = self.main_gui.get_border_padding()
        x, y = self.pad_coordinates(
            properties.get_member_by_path(event, event_type['x_path']),
            properties.get_member_by_path(event, event_type['y_path']),
            padded_x,
            padded_y,
        )
        self.selected_event_group.setPos(16 * x, 16 * y)

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
            if item.pixmap().cacheKey() != pixmap.cacheKey():
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
        brush = QBrush(border_color)
        no_border_pen = QPen(Qt.PenStyle.NoPen)
        # no outline
        self.north_border = self.addRect(
            0,
            0,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            brush=brush,
            pen=no_border_pen,
        )
        self.south_border = self.addRect(
            0,
            (padded_height + map_height) * 16,
            (2 * padded_width + map_width) * 16,
            padded_height * 16,
            brush=brush,
            pen=no_border_pen,
        )
        self.west_border = self.addRect(
            0,
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            brush=brush,
            pen=no_border_pen,
        )
        self.east_border = self.addRect(
            16 * (padded_width + map_width),
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            brush=brush,
            pen=no_border_pen,
        )
        self.border_effect_group.addToGroup(self.north_border)
        self.border_effect_group.addToGroup(self.south_border)
        self.border_effect_group.addToGroup(self.west_border)
        self.border_effect_group.addToGroup(self.east_border)
        self.addItem(self.border_effect_group)
        self._group_set_flags(self.border_effect_group)

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
        self.visible_layers = visible_layers
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
        if self.selected_event_group is not None:
            self.selected_event_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.SELECTED_EVENT) > 0
            )
        if self.connection_rectangles_group is not None:
            self.connection_rectangles_group.setVisible(
                (visible_layers & MapScene.VisibleLayer.CONNECTION_RECTANGLES) > 0
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

    @staticmethod
    def fix_rectangle(
        x: int, y: int, width: int, height: int, max_width: int, max_height: int
    ) -> tuple[int, int, int, int]:
        """Fixes the position of a rectangle to fit into the graphics scene."""
        # Fix negative bounds
        x, width = max(0, x), width + min(0, x)
        y, height = max(0, y), height + min(0, y)
        # Fix positive bounds
        if x + width > max_width:
            width = max_width - x
        if y + height > max_height:
            height = max_height - y
        # If width or height became negative, do not show the rect
        width, height = max(0, width), max(0, height)
        return x, y, width, height
