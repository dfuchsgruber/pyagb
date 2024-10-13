"""Widget for the connection map scene."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.gui.map_scene import MapScene as BaseMapScene
from pymap.gui.types import ConnectionType, UnpackedConnection

from .. import history

if TYPE_CHECKING:
    from .connection_widget import ConnectionWidget


class MapScene(BaseMapScene):
    """Scene for the map view."""

    def __init__(
        self,
        connection_widget: ConnectionWidget,
        parent: QWidget | None = None,
    ):
        """Initializes the map scene.

        Args:
            connection_widget (ConnectionWidget): The connection widget.
            parent (QtWidgets.QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(connection_widget.main_gui, parent=parent)
        self.connection_widget = connection_widget

        # Store the position where a drag happend recently so there are not multiple
        # draw events per block
        self.last_drag = None
        # Store the index of the connection that is dragged
        self.dragged_idx = -1
        # Indicate if the connection has at least moved one block,
        # i.e. if a macro is currently active
        self.drag_started = False
        # Store the position where the drag originated in order to calculate an offset
        self.drag_origin = None

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for hover events on the map image."""
        if not self.connection_widget.connection_loaded:
            return
        assert self.connection_widget.main_gui.project is not None
        map_width, map_height = self.connection_widget.main_gui.get_map_dimensions()

        padded_x, padded_y = self.connection_widget.main_gui.project.config['pymap'][
            'display'
        ]['border_padding']

        pos = event.scenePos()

        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(map_width + 2 * padded_x) and y in range(
            map_height + 2 * padded_y
        ):
            if x - padded_x in range(map_width) and y - padded_y in range(map_height):
                self.connection_widget.info_label.setText(
                    f'x : {hex(x - padded_x)}, y : {hex(y - padded_y)}'
                )
            else:
                self.connection_widget.info_label.setText('')

            if self.last_drag is not None and self.last_drag != (x, y):
                # Start the dragging macro if not started already

                if not self.drag_started:
                    self.drag_started = True

                    self.connection_widget.undo_stack.beginMacro('Drag event')

                # Drag the connection to this position

                connection = self.connection_widget.connections[self.dragged_idx]
                assert connection is not None

                if connection.type in (ConnectionType.NORTH, ConnectionType.SOUTH):
                    offset_new = connection.offset + x - self.last_drag[0]
                elif connection.type in (ConnectionType.EAST, ConnectionType.WEST):
                    offset_new = connection.offset + y - self.last_drag[1]
                else:
                    raise ValueError(
                        f'Invalid connection type {connection.type} for dragging.'
                    )

                self.connection_widget.connections[
                    self.dragged_idx
                ] = UnpackedConnection(
                    connection.type,
                    offset_new,
                    connection.bank,
                    connection.map_idx,
                    connection.blocks,
                )

                statement_redo, statement_undo = history.path_to_statement(
                    self.connection_widget.main_gui.project.config['pymap']['header'][
                        'connections'
                    ]['connection_offset_path'],
                    connection.offset,
                    offset_new,
                )

                self.connection_widget.undo_stack.push(
                    history.ChangeConnectionProperty(
                        self.connection_widget,
                        self.dragged_idx,
                        self.connection_widget.mirror_offset.isChecked(),
                        [statement_redo],
                        [statement_undo],
                    )
                )

                self.last_drag = x, y

        else:
            self.connection_widget.info_label.setText('')

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.connection_widget.connection_loaded:
            return
        assert self.connection_widget.main_gui.project is not None

        map_width, map_height = self.connection_widget.main_gui.get_map_dimensions()

        padded_x, padded_y = self.connection_widget.main_gui.project.config['pymap'][
            'display'
        ]['border_padding']

        pos = event.scenePos()

        x, y = int(pos.x() / 16), int(pos.y() / 16)

        if (
            x in range(2 * padded_x + map_width)
            and y in range(2 * padded_y + map_height)
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # Check if there is any connection

            self.dragged_idx = -1
            for direction in self.connection_widget.connection_rects:
                rect = self.connection_widget.connection_rects[direction].rect()

                if (
                    x >= int(rect.x() / 16)
                    and x < int((rect.x() + rect.width()) / 16)
                    and y >= int(rect.y() / 16)
                    and y < int((rect.y() + rect.height()) / 16)
                ):
                    # Find the index that matches direction

                    for idx, connection in enumerate(
                        self.connection_widget.connections
                    ):
                        if connection is not None and direction == connection[0]:
                            self.dragged_idx = idx

                            break

                    if self.dragged_idx == -1:
                        raise RuntimeError(
                            f'Inconsistent connection type {direction}. '
                            ' Did not find any matching connection for a rectangle.'
                        )

                    self.connection_widget.idx_combobox.setCurrentIndex(
                        self.dragged_idx
                    )
                    self.last_drag = x, y
                    self.drag_origin = x, y
                    self.drag_started = False

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if not self.connection_widget.connection_loaded:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drag_started:
                # End a macro only if the event has at least moved one block
                self.connection_widget.undo_stack.endMacro()

            self.drag_started = False
            self.dragged_idx = -1
            self.last_drag = None
            self.drag_origin = None

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Double click event handler for the map image."""
        if not self.connection_widget.connection_loaded:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if (
                self.connection_widget.main_gui.project is None
                or self.connection_widget.main_gui.header is None
            ):
                return
            map_width, map_height = self.connection_widget.main_gui.get_map_dimensions()
            padded_x, padded_y = self.connection_widget.main_gui.project.config[
                'pymap'
            ]['display']['border_padding']

            pos = event.scenePos()

            x, y = int(pos.x() / 16), int(pos.y() / 16)

            if (
                x in range(2 * padded_x + map_width)
                and y in range(2 * padded_y + map_height)
                and event.button() == Qt.MouseButton.LeftButton
            ):
                # Check if there is any connection

                self.dragged_idx = -1

                for direction in self.connection_widget.connection_rects:
                    rect = self.connection_widget.connection_rects[direction].rect()

                    if (
                        x >= int(rect.x() / 16)
                        and x < int((rect.x() + rect.width()) / 16)
                        and y >= int(rect.y() / 16)
                        and y < int((rect.y() + rect.height()) / 16)
                    ):
                        # Find the index that matches direction

                        for _, connection in enumerate(
                            self.connection_widget.connections
                        ):
                            if connection is not None and direction == connection.type:
                                self.connection_widget.main_gui.open_header(
                                    str(connection.bank), str(connection.map_idx)
                                )
                                break
