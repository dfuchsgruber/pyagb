"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QMessageBox,
    QWidget,
)

from agb.model.type import ModelValue
from pymap.gui.blocks import (
    connection_get_bank,
    connection_get_connection_type,
    connection_get_map_idx,
    connection_get_offset,
    unpack_connection,
)
from pymap.gui.history.connection import (
    AppendConnection,
    ChangeConnectionProperty,
    RemoveConnection,
)
from pymap.gui.history.statement import path_to_statement
from pymap.gui.icon import Icon, icon_paths
from pymap.gui.map.view import VisibleLayer
from pymap.gui.properties.utils import get_member_by_path, set_member_by_path
from pymap.gui.types import ConnectionType, opposite_connection_direction

# from pymap.debug import Profile
from ..tab import MapWidgetTab
from .properties import ConnectionProperties

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class ConnectionsTab(MapWidgetTab):
    """Tab for the events on the map."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

        # Layout is similar to the map widget
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.mirror_offset = QtWidgets.QCheckBox('Mirror Displacement to Adjacent Map')
        self.mirror_offset.setChecked(
            self.map_widget.main_gui.settings.value(
                'connections/mirror_offset', True, bool
            )  # type: ignore
        )
        self.mirror_offset.stateChanged.connect(self.mirror_offset_changed)
        self.idx_combobox = QtWidgets.QComboBox()
        self.add_button = QtWidgets.QPushButton()
        self.add_button.setIcon(QIcon(icon_paths[Icon.PLUS]))
        self.add_button.clicked.connect(self.append_connection)
        self.remove_button = QtWidgets.QPushButton()
        self.remove_button.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        self.remove_button.clicked.connect(
            lambda: self.remove_connection(self.idx_combobox.currentIndex())
        )
        self.connection_properties = ConnectionProperties(self)
        self.open_connection = QtWidgets.QPushButton('Open adjacent map')
        self.open_connection.clicked.connect(self.open_adjacent_map)
        self.idx_combobox.currentIndexChanged.connect(self.select_connection)

        layout.addWidget(self.mirror_offset, 1, 1, 1, 3)
        layout.addWidget(self.idx_combobox, 2, 1)
        layout.addWidget(self.open_connection, 4, 1, 1, 3)
        layout.addWidget(self.connection_properties, 3, 1, 1, 3)
        layout.addWidget(self.remove_button, 2, 3)
        layout.addWidget(self.add_button, 2, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        # Recording mouse events
        self.last_dragged_position: tuple[int, int] | None = None
        self.dragged_connection_idx = None
        self.is_dragging = False

    @property
    def connection_loaded(self) -> bool:
        """Returns whether a connection is loaded."""
        return (
            self.map_widget.main_gui.project is not None
            and self.map_widget.main_gui.header is not None
        )

    def mirror_offset_changed(self):
        """Event handler for when the mirror offset checkbox is toggled."""
        self.map_widget.main_gui.settings.setValue(
            'connections/mirror_offset', self.mirror_offset.isChecked()
        )

    def load_project(self):
        """Loads the project."""
        pass

    def load_header(self):
        """Loads a header."""
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
            or self.map_widget.main_gui.footer is None
        ):
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
            return

        connections_model = self.map_widget.main_gui.get_connections()
        self.idx_combobox.blockSignals(True)
        self.idx_combobox.clear()
        self.idx_combobox.addItems(list(map(str, range(len(connections_model)))))
        self.idx_combobox.setCurrentIndex(self.selected_connection_idx)
        # We want select connection to be triggered even if the current idx
        # is -1 in order to clear the properties
        self.select_connection(self.selected_connection_idx)
        self.idx_combobox.blockSignals(False)

    @property
    def visible_layers(self) -> VisibleLayer:
        """Get the visible layers."""
        return (
            VisibleLayer.BLOCKS
            | VisibleLayer.BORDER_EFFECT
            | VisibleLayer.CONNECTION_RECTANGLES
            | VisibleLayer.GRID
        )

    @property
    def selected_connection_idx(self) -> int:
        """Returns the index of currently selected connection or -1.

        If no connections exist, returns -1. If the combox has selected index -1,
        returns 0.
        """
        return min(
            len(self.map_widget.main_gui.get_connections()) - 1,
            max(0, self.idx_combobox.currentIndex()),
        )

    def _get_connection_by_padded_position(
        self,
        x: int,
        y: int,
    ) -> int | None:
        """Gets the event associated with a mouse position."""
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
        ):
            return None
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        padded_x, padded_y = self.map_widget.main_gui.get_border_padding()

        if x in range(2 * padded_x + map_width) and y in range(
            2 * padded_y + map_height
        ):
            connections = self.map_widget.main_gui.get_connections()
            for connection_idx in range(len(connections)):
                # Get the position of the connection rectangle
                rectangle_x, rectangle_y, rectangle_width, rectangle_height = (
                    self.map_widget.map_scene_view.connections.connection_rectangle_get_position_and_dimensions(
                        connection_idx
                    )
                )
                if x in range(
                    rectangle_x // 16, (rectangle_x + rectangle_width) // 16
                ) and y in range(
                    rectangle_y // 16, (rectangle_y + rectangle_height) // 16
                ):
                    return connection_idx

        return None

    def _select_connection_at_padded_position(
        self,
        x: int,
        y: int,
    ):
        """Selects an event at the given position."""
        target = self._get_connection_by_padded_position(x, y)
        if target is not None:
            if self.idx_combobox.currentIndex() != target:
                self.idx_combobox.setCurrentIndex(target)
            self.last_dragged_position = x, y
            self.dragged_connection_idx = target
            self.is_dragging = False
        else:
            # If no event is selected, set the dragged event to None
            self.last_dragged_position = None
            self.dragged_connection_idx = None
            self.is_dragging = False

    def map_scene_mouse_pressed(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if not event.button() == Qt.MouseButton.LeftButton:
            return
        self._select_connection_at_padded_position(x, y)

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_connection_at_padded_position(x, y)

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_connection_at_padded_position(x, y)

    def map_scene_mouse_moved(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        assert self.map_widget.main_gui.project is not None
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        if self.last_dragged_position is not None and self.last_dragged_position != (
            x,
            y,
        ):
            assert self.dragged_connection_idx is not None, 'Dragged event is None'
            if not self.is_dragging:
                self.is_dragging = True
                self.map_widget.undo_stack.beginMacro('Drag Connection')
            connection = self.map_widget.main_gui.get_connections()[
                self.dragged_connection_idx
            ]
            connection_type = connection_get_connection_type(
                connection, self.map_widget.main_gui.project
            )
            offset = connection_get_offset(connection, self.map_widget.main_gui.project)
            if offset is not None:
                if connection_type in (ConnectionType.NORTH, ConnectionType.SOUTH):
                    offset_new = offset + x - self.last_dragged_position[0]
                elif connection_type in (ConnectionType.EAST, ConnectionType.WEST):
                    offset_new = offset + y - self.last_dragged_position[1]
                else:
                    raise ValueError(
                        f'Invalid connection type {connection_type} for dragging.'
                    )
                statement_redo, statement_undo = path_to_statement(
                    self.map_widget.main_gui.project.config['pymap']['header'][
                        'connections'
                    ]['connection_offset_path'],
                    offset,
                    offset_new,
                )
                self.map_widget.undo_stack.push(
                    ChangeConnectionProperty(
                        self,
                        self.dragged_connection_idx,
                        self.mirror_offset.isChecked(),
                        [statement_redo],
                        [statement_undo],
                    )
                )

            self.last_dragged_position = x, y

    def map_scene_mouse_released(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if self.is_dragging:
            self.is_dragging = False
            self.map_widget.undo_stack.endMacro()
        self.last_dragged_position = None
        self.dragged_connection_idx = None

    def map_scene_mouse_double_clicked(
        self, event: QMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for double clicking the mouse."""
        if not self.map_widget.header_loaded:
            return
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        target = self._get_connection_by_padded_position(x, y)
        if target is None:
            return
        self.open_adjacent_map_of_connection(target)

    def append_connection(self) -> None:
        """Append a new connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        self.map_widget.undo_stack.push(AppendConnection(self))

    def remove_connection(self, index: int) -> None:
        """Remove a connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        if index < 0:
            return
        self.map_widget.undo_stack.push(RemoveConnection(self, index))

    def open_adjacent_map(self) -> None:
        """Open the adjacent map."""
        idx = self.idx_combobox.currentIndex()
        if idx < 0:
            return
        self.open_adjacent_map_of_connection(idx)

    def open_adjacent_map_of_connection(self, idx: int):
        """Open the adjacent map of a connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        assert self.map_widget.main_gui.project is not None
        connection = self.map_widget.main_gui.get_connections()[idx]
        bank = connection_get_bank(connection, self.map_widget.main_gui.project)
        map_idx = connection_get_map_idx(connection, self.map_widget.main_gui.project)
        if bank is None or map_idx is None:
            return
        # Parse bank and map_idx
        try:
            bank = str(int(str(bank), 0))
        except ValueError:
            bank = str(bank)
        try:
            map_idx = str(int(str(map_idx), 0))
        except ValueError:
            map_idx = str(map_idx)
        try:
            self.map_widget.main_gui.open_header(bank, map_idx)
        except KeyError as e:
            QMessageBox.critical(
                self,
                'Header can not be opened',
                f'The header {bank}.{map_idx}'
                ' could not be opened.'
                f' Key {e.args[0]} was not found.',
            )
            return

    def select_connection(self, idx: int) -> None:
        """Select a connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        self.connection_properties.load()
        self.map_widget.map_scene_view.connections.update_selected_connection(
            self.selected_connection_idx
        )

    def update_connection(
        self,
        connection_idx: int,
        previous_value: ModelValue | None = None,
    ):
        """Updates a connection.

        It reloads the connection's block and re-computes all blocks and sets the
        graphics.
        """
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        connections = self.map_widget.main_gui.get_connections()
        connection = connections[connection_idx]

        if previous_value is not None:
            connection_type_previous = connection_get_connection_type(
                previous_value, self.map_widget.main_gui.project
            )
            connection_type = connection_get_connection_type(
                connection, self.map_widget.main_gui.project
            )
            bank_previous = connection_get_bank(
                previous_value, self.map_widget.main_gui.project
            )
            bank = connection_get_bank(connection, self.map_widget.main_gui.project)
            map_idx_previous = connection_get_map_idx(
                previous_value,
                self.map_widget.main_gui.project,
            )
            map_idx = connection_get_map_idx(
                connection, self.map_widget.main_gui.project
            )
            reload_blocks = (
                bank != bank_previous
                or map_idx != map_idx_previous
                or connection_type != connection_type_previous
            )
        else:
            reload_blocks = True

        if reload_blocks:
            connections[connection_idx] = unpack_connection(
                connection, self.map_widget.main_gui.project
            )

    def mirror_connection_update_to_adjacent_connection(self, connection_idx: int):
        """Mirror the connection update to the adjacent connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        connections = self.map_widget.main_gui.get_connections()
        connection = connections[connection_idx]
        connection_type = connection_get_connection_type(
            connection, self.map_widget.main_gui.project
        )
        connection_offset = connection_get_offset(
            connection, self.map_widget.main_gui.project
        )
        bank = connection_get_bank(connection, self.map_widget.main_gui.project)
        map_idx = connection_get_map_idx(connection, self.map_widget.main_gui.project)
        if (
            bank is None
            or map_idx is None
            or connection_type is None
            or connection_offset is None
        ):
            return
        header, _, _ = self.map_widget.main_gui.project.load_header(
            bank, map_idx, unpack_connections=False
        )
        if header is None:
            return
        # Find the connection in the adjacent header
        connections_adjacent = get_member_by_path(
            header,
            self.map_widget.main_gui.project.config['pymap']['header']['connections'][
                'connections_path'
            ],
        )
        assert isinstance(connections_adjacent, list), (
            'Connections adjacent is not a list'
        )
        for connection_idx, connection_adjacent in enumerate(connections_adjacent):
            bank_adjacent = connection_get_bank(
                connection_adjacent, self.map_widget.main_gui.project
            )
            map_idx_adjacent = connection_get_map_idx(
                connection_adjacent, self.map_widget.main_gui.project
            )
            # Parse bank and map_idx
            try:
                bank_adjacent = str(int(str(bank_adjacent), 0))
            except ValueError:
                bank_adjacent = str(bank_adjacent)
            try:
                map_idx_adjacent = str(int(str(map_idx_adjacent), 0))
            except ValueError:
                map_idx_adjacent = str(map_idx_adjacent)

            connection_type_adjacent = connection_get_connection_type(
                connection_adjacent, self.map_widget.main_gui.project
            )

            if (
                bank_adjacent == self.map_widget.main_gui.header_bank
                and map_idx_adjacent == self.map_widget.main_gui.header_map_idx
                and opposite_connection_direction[connection_type]
                == connection_type_adjacent
            ):
                # Update the connection in the adjacent header
                break
        else:
            return
        # Change the offset in the adjacent connection
        set_member_by_path(
            connection_adjacent,
            str(-connection_offset),
            self.map_widget.main_gui.project.config['pymap']['header']['connections'][
                'connection_offset_path'
            ],
        )
        self.map_widget.main_gui.project.save_header(
            header, bank, map_idx, pack_connections=False
        )
