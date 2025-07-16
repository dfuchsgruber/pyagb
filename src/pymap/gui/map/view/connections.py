"""Handles the connection rectangles in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsRectItem,
)

from pymap.gui.blocks import (
    connection_get_blocks,
    connection_get_connection_type,
    connection_get_offset,
)
from pymap.gui.types import ConnectionType, visible_connection_directions

from .layer import MapViewLayer

if TYPE_CHECKING:
    pass


class MapViewLayerConnections(MapViewLayer):
    """A layer in the map view that displays connection rectangles."""

    def load_map(self) -> None:
        """Loads the connection rectangles into the scene."""
        assert (
            self.view.main_gui.project is not None
            and self.view.main_gui.header is not None
        )
        connection_rectangles_group = QGraphicsItemGroup()
        connection_color = QColor.fromRgbF(
            *(self.view.main_gui.project.config['pymap']['display']['connection_color'])
        )
        connection_border_color = QColor.fromRgbF(
            *(
                self.view.main_gui.project.config['pymap']['display'][
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
            self._connection_rectangles[connection_type] = QGraphicsRectItem(
                *rect,
            )
            self._connection_rectangles[connection_type].setPen(
                QPen(connection_border_color)
            )
            self._connection_rectangles[connection_type].setBrush(
                QBrush(connection_color)
            )
            connection_rectangles_group.addToGroup(
                self._connection_rectangles[connection_type]
            )
        self.item = connection_rectangles_group

    @property
    def visible_connection_idxs(self) -> dict[ConnectionType, int]:
        """Returns the visible connection indexes."""
        assert self.view.main_gui.project is not None
        visible_connections = {
            connection_type: next(
                (
                    idx
                    for idx, connection in enumerate(
                        self.view.main_gui.get_connections()
                    )
                    if connection is not None
                    and str(
                        connection_get_connection_type(
                            connection, self.view.main_gui.project
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
        assert self.view.main_gui.project is not None
        padded_width, padded_height = self.view.main_gui.get_border_padding()
        map_width, map_height = self.view.main_gui.get_map_dimensions()
        connections = self.view.main_gui.get_connections()
        connection = connections[connection_idx]
        if connection is None:
            return 0, 0, 0, 0
        connection_blocks = connection_get_blocks(
            connection,
            self.view.main_gui.project,
        )
        connection_type = connection_get_connection_type(
            connection,
            self.view.main_gui.project,
        )
        connection_offset = connection_get_offset(
            connection,
            self.view.main_gui.project,
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

        return self.view.fix_rectangle(
            *rect,
            16 * (map_width + 2 * padded_width),
            16 * (map_height + 2 * padded_height),
        )

    def update_connection_rectangles(self):
        """Updates the connection rectangles."""
        assert self.view.main_gui.project is not None
        if self.item is None:
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
        assert self.view.main_gui.project is not None
        if self.item is None or connection_idx < 0:
            return
        connection_color = QColor.fromRgbF(
            *(self.view.main_gui.project.config['pymap']['display']['connection_color'])
        )
        connection_active_color = QColor.fromRgbF(
            *(
                self.view.main_gui.project.config['pymap']['display'][
                    'connection_active_color'
                ]
            )
        )
        connection_border_color = QColor.fromRgbF(
            *(
                self.view.main_gui.project.config['pymap']['display'][
                    'connection_border_color'
                ]
            )
        )
        connection_active_border_color = QColor.fromRgbF(
            *(
                self.view.main_gui.project.config['pymap']['display'][
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
