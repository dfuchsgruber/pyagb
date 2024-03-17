"""Widget for connection editing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageQt import ImageQt
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QMessageBox

from pymap.gui import blocks
from pymap.gui.icon import Icon, icon_paths
from pymap.gui.types import UnpackedConnection, ConnectionType

from .. import history, properties
from .connection_properties import ConnectionProperties
from .map_scene import MapScene

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class ConnectionWidget(QtWidgets.QWidget):
    """Widget to edit connections."""

    def __init__(self, main_gui: PymapGui, parent: QtWidgets.QWidget | None = None):
        """Initializes the connection widget.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QtWidgets.QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.connections: list[UnpackedConnection | None] = []
        self.undo_stack = QtGui.QUndoStack()

        # Layout is similar to the map widget

        layout = QtWidgets.QGridLayout()

        self.setLayout(layout)

        splitter = QtWidgets.QSplitter()

        layout.addWidget(splitter, 1, 1, 1, 1)

        self.map_scene = MapScene(self)

        self.map_scene_view = QtWidgets.QGraphicsView()
        self.map_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.map_scene_view.setScene(self.map_scene)
        splitter.addWidget(self.map_scene_view)
        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label, 2, 1, 1, 1)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        self.connection_widget = QtWidgets.QWidget()
        splitter.addWidget(self.connection_widget)
        splitter.setSizes([4 * 10**6, 10**6])  # Ugly as hell hack to take large values
        connection_layout = QtWidgets.QGridLayout()
        self.connection_widget.setLayout(connection_layout)
        self.mirror_offset = QtWidgets.QCheckBox('Mirror Displacement to Adjacent Map')
        self.mirror_offset.setChecked(
            self.main_gui.settings.settings['connections_mirror_offset']
        )
        self.mirror_offset.stateChanged.connect(self.mirror_offset_changed)
        connection_layout.addWidget(self.mirror_offset, 1, 1, 1, 3)
        self.idx_combobox = QtWidgets.QComboBox()
        connection_layout.addWidget(self.idx_combobox, 2, 1)
        self.add_button = QtWidgets.QPushButton()
        self.add_button.setIcon(QtGui.QIcon(icon_paths[Icon.PLUS]))
        self.add_button.clicked.connect(self.append_connection)
        connection_layout.addWidget(self.add_button, 2, 2)
        self.remove_button = QtWidgets.QPushButton()
        self.remove_button.setIcon(QtGui.QIcon(icon_paths[Icon.REMOVE]))
        self.remove_button.clicked.connect(
            lambda: self.remove_connection(self.idx_combobox.currentIndex())
        )
        connection_layout.addWidget(self.remove_button, 2, 3)
        self.connection_properties = ConnectionProperties(self)
        connection_layout.addWidget(self.connection_properties, 3, 1, 1, 3)
        self.open_connection = QtWidgets.QPushButton('Open adjacent map')
        connection_layout.addWidget(self.open_connection, 4, 1, 1, 3)
        self.open_connection.clicked.connect(self.open_adjacent_map)
        connection_layout.setColumnStretch(1, 1)
        connection_layout.setColumnStretch(2, 0)
        connection_layout.setColumnStretch(3, 0)
        self.idx_combobox.currentIndexChanged.connect(self.select_connection)

    @property
    def connection_loaded(self) -> bool:
        """Returns whether a connection is loaded."""
        return self.main_gui.project is not None and self.main_gui.header is not None

    def mirror_offset_changed(self):
        """Event handler for when the mirror offset checkbox is toggled."""
        self.main_gui.settings.settings[
            'connections_mirror_offset'
        ] = self.mirror_offset.isChecked()

    def load_project(self):
        """Loads a new project."""
        self.load_header()

    def load_header(self):
        """Loads graphics for the current header."""
        self.map_scene.clear()

        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
        ):
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
            return

        self.base_blocks = blocks.compute_blocks(
            self.main_gui.footer, self.main_gui.project
        )

        # Load connections
        connections_model = self.main_gui.get_connections()
        self.connections = blocks.unpack_connections(
            connections_model,  # type: ignore
            self.main_gui.project,
        )
        assert self.main_gui.blocks is not None

        # Load the current blocks
        self.blocks = self.compute_blocks()
        self.block_pixmaps = np.empty_like(self.blocks[:, :, 0], dtype=object)
        for (y, x), block_idx in np.ndenumerate(self.blocks[:, :, 0]):
            # Draw the blocks
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
            self.block_pixmaps[y, x] = item

        padded_width, padded_height = self.main_gui.project.config['pymap']['display'][
            'border_padding'
        ]
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Draw rectangles for the borders
        border_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['border_color'])
        )
        self.map_scene.addRect(
            0,
            0,
            16 * (map_width + 2 * padded_width),
            16 * padded_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )

        self.map_scene.addRect(
            0,
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )

        self.map_scene.addRect(
            16 * (padded_width + map_width),
            16 * padded_height,
            16 * padded_width,
            16 * map_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )

        self.map_scene.addRect(
            0,
            16 * (padded_height + map_height),
            16 * (map_width + 2 * padded_width),
            16 * padded_height,
            pen=QPen(0),
            brush=QBrush(border_color),
        )

        # Setup zero size rectangles for all possible connections

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

        self.connection_rects = {
            direction: self.map_scene.addRect(
                0,
                0,
                0,
                0,
                pen=QPen(connection_border_color),
                brush=QBrush(connection_color),
            )
            for direction in ConnectionType
        }

        self.update_border_rectangles()

        self.map_scene.setSceneRect(
            0,
            0,
            16 * (map_width + 2 * padded_width),
            16 * (map_height + 2 * padded_height),
        )

        current_idx = min(
            len(self.connections) - 1, max(0, self.idx_combobox.currentIndex())
        )  # If -1 is selcted, select first, but never select a no more present event

        self.idx_combobox.blockSignals(True)
        self.idx_combobox.clear()
        self.idx_combobox.addItems(list(map(str, range(len(self.connections)))))
        self.idx_combobox.setCurrentIndex(current_idx)
        # We want select connection to be triggered even if the current idx
        # is -1 in order to clear the properties
        self.select_connection()
        self.idx_combobox.blockSignals(False)

    def select_connection(self):
        """Selects the event of the current index."""
        self.connection_properties.load_connection()
        self.update_border_rectangles()

    def compute_blocks(self):
        """Comptues the current block map with the current connections."""
        map_blocks = self.base_blocks.copy()
        assert self.main_gui.project is not None
        for connection in blocks.filter_visible_connections(self.connections):
            blocks.insert_connection(
                map_blocks, connection, self.main_gui.footer, self.main_gui.project
            )
        return map_blocks

    def remove_connection(self, connection_idx: int):
        """Removes a connection."""
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        if connection_idx < 0:
            return
        self.undo_stack.push(history.RemoveConnection(self, connection_idx))

    def append_connection(self):
        """Appends a new connection."""
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.undo_stack.push(history.AppendConnection(self))

    def open_adjacent_map(self):
        """Opens the currently selected adjacent map."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.idx_combobox.currentIndex() < 0
        ):
            return

        connections_model = self.main_gui.get_connections()
        assert isinstance(
            connections_model, list
        ), f'Expected list, got {type(connections_model)}'
        packed = connections_model[self.idx_combobox.currentIndex()]
        assert isinstance(
            packed, UnpackedConnection
        ), f'Expected Connection, got {type(packed)}'

        # Update the unpacked version

        connection = blocks.unpack_connection(
            packed, self.main_gui.project, connection_blocks=None
        )
        assert connection is not None

        try:
            self.main_gui.open_header(str(connection.bank), str(connection.map_idx))
        except KeyError as e:
            return QMessageBox.critical(
                self,
                'Header can not be opened',
                f'The header {connection.bank}.{connection.map_idx} could not be opened.'
                f' Key {e.args[0]} was not found.',
            )

    def update_connection(self, connection_idx: int, mirror_offset: bool):
        """Updates a connection."""
        assert self.main_gui.project is not None
        if self.idx_combobox.currentIndex() == connection_idx:
            self.connection_properties.update()

        connections_model = self.main_gui.get_connections()
        packed = connections_model[connection_idx]
        assert isinstance(
            packed, UnpackedConnection
        ), f'Expected Connection, got {type(packed)}'

        # Update the unpacked version
        self.connections[connection_idx] = blocks.unpack_connection(
            packed, self.main_gui.project, connection_blocks=None
        )
        self.update_blocks()
        self.update_border_rectangles()

        # Mirror offset changes to the adjacent map
        connection = self.connections[connection_idx]
        if mirror_offset and connection is not None:
            # Load the adjacent header
            header, _, _ = self.main_gui.project.load_header(
                connection.bank, connection.map_idx
            )

            if header is not None:
                # Find the correlating connection
                adjacent_packed = self.main_gui.get_connections()
                adjacent_connections = blocks.unpack_connections(
                    adjacent_packed,
                    self.main_gui.project,
                    default_blocks=np.empty((0, 0, 2), dtype=int),
                )

                for idx, adjacent_connection in enumerate(adjacent_connections):
                    # Check if the adjacent bank and map idx match the current map
                    if adjacent_connection is None:
                        continue

                    # Bring bank and map_idx in their canonical forms
                    try:
                        adjacent_map_idx = str(int(str(adjacent_connection.map_idx), 0))
                    except ValueError:
                        adjacent_map_idx = str(adjacent_connection.map_idx)

                    try:
                        adjacent_bank = str(int(str(adjacent_connection.bank), 0))
                    except ValueError:
                        adjacent_bank = str(adjacent_connection.bank)

                    if (
                        opposite_directions[ConnectionType(connection.type)]
                        == adjacent_connection.type
                        and adjacent_bank == self.main_gui.header_bank
                        and adjacent_map_idx == self.main_gui.header_map_idx
                    ):
                        # Match, mirror the change
                        assert isinstance(
                            adjacent_packed, list
                        ), f'Expected list, got {type(adjacent_packed)}'
                        properties.set_member_by_path(
                            adjacent_packed[idx],
                            str(-connection.offset),
                            self.main_gui.project.config['pymap']['header'][
                                'connections'
                            ]['connection_offset_path'],
                        )

                        self.main_gui.project.save_header(
                            header, connection.bank, connection.map_idx
                        )

    def update_blocks(self):
        """Visually updates the blocks and connections."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.blocks is None
        ):
            return
        new_blocks = self.compute_blocks()

        # Check which blocks have changed and only render those

        for y, x in zip(*np.where(new_blocks[:, :, 0] != self.blocks[:, :, 0])):
            assert isinstance(y, int), f'Expected int, got {type(y)}'
            assert isinstance(x, int), f'Expected int, got {type(x)}'
            block_idx = new_blocks[y, x, 0]
            assert isinstance(block_idx, int), f'Expected int, got {type(block_idx)}'
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            self.block_pixmaps[y, x].setPixmap(pixmap)

        self.blocks = new_blocks

    def update_border_rectangles(self):
        """Updates the border rectangles."""
        assert self.main_gui.project is not None
        # First hide all rectangles
        for direction in self.connection_rects:
            self.connection_rects[direction].setRect(0, 0, 0, 0)

        # Show border rectangles

        padded_width, padded_height = self.main_gui.project.config['pymap']['display'][
            'border_padding'
        ]
        map_width, map_height = self.main_gui.get_map_dimensions()

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

        for idx, connection in enumerate(
            blocks.filter_visible_connections(self.connections, keep_invisble=True)
        ):
            if connection is None:
                continue

            connection_width, connection_height = (
                connection.blocks.shape[1],
                connection.blocks.shape[0],
            )

            match connection.type:
                case ConnectionType.NORTH:
                    rect = (
                        16 * (padded_width + connection.offset),
                        16 * (padded_height - connection_height),
                        16 * connection_width,
                        16 * connection_height,
                    )
                case ConnectionType.SOUTH:
                    rect = (
                        16 * (padded_width + connection.offset),
                        16 * (padded_height + map_height),
                        16 * connection_width,
                        16 * connection_height,
                    )
                case ConnectionType.EAST:
                    rect = (
                        16 * (padded_width + map_width),
                        16 * (padded_height + connection.offset),
                        16 * connection_width,
                        16 * connection_height,
                    )
                case ConnectionType.WEST:
                    rect = (
                        16 * (padded_width - connection_width),
                        16 * (padded_height + connection.offset),
                        16 * connection_width,
                        16 * connection_height,
                    )
                case _:
                    raise ValueError(f'Invalid connection type {connection.type}')

            self.connection_rects[connection.type].setRect(
                *fix_rect(
                    *rect,
                    16 * (map_width + 2 * padded_width),
                    16 * (map_height + 2 * padded_height),
                )
            )

            if idx == self.idx_combobox.currentIndex():
                self.connection_rects[connection.type].setPen(
                    QPen(connection_active_border_color)
                )

                self.connection_rects[connection.type].setBrush(
                    QBrush(connection_active_color)
                )

            else:
                self.connection_rects[connection.type].setPen(
                    QPen(connection_border_color)
                )

                self.connection_rects[connection.type].setBrush(
                    QBrush(connection_color)
                )

        self.map_scene.setSceneRect(
            0,
            0,
            16 * (map_width + 2 * padded_width),
            16 * (map_height + 2 * padded_height),
        )


def fix_rect(
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


opposite_directions: dict[ConnectionType, ConnectionType] = {
    ConnectionType.NORTH: ConnectionType.SOUTH,
    ConnectionType.SOUTH: ConnectionType.NORTH,
    ConnectionType.EAST: ConnectionType.WEST,
    ConnectionType.WEST: ConnectionType.EAST,
}
