"""History actions for map headers."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelValue
from pymap.gui.blocks import unpack_connection
from pymap.gui.history.statement import ChangeProperty, UndoRedoStatements
from pymap.gui.properties import get_parents_by_path

if TYPE_CHECKING:
    from pymap.gui.map.tabs.connections.connections import ConnectionsTab


class ChangeConnectionProperty(ChangeProperty):
    """Change a property of any vent."""

    def __init__(
        self,
        connections_tab: ConnectionsTab,
        connection_idx: int,
        mirror_offset: bool,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the event property change.

        Args:
            connections_tab (ConnectionsTab): reference to the connection widget
            connection_idx (int): index of the connection
            mirror_offset (bool): whether the connection is mirrored
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__(
            statements_redo, statements_undo, text='Change Connection Property'
        )
        self.connections_tab = connections_tab
        self.connection_idx = connection_idx
        self.mirror_offset = mirror_offset
        self.previous_value = deepcopy(self.get_root())

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        connections = self.connections_tab.map_widget.main_gui.get_connections()
        return connections[self.connection_idx]

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.connections_tab.update_connection(
            self.connection_idx, previous_value=self.previous_value
        )
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()
        if self.mirror_offset:
            self.connections_tab.mirror_connection_update_to_adjacent_connection(
                self.connection_idx,
            )

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.connections_tab.update_connection(
            self.connection_idx, previous_value=self.previous_value
        )
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()
        if self.mirror_offset:
            self.connections_tab.mirror_connection_update_to_adjacent_connection(
                self.connection_idx,
            )


class AppendConnection(QUndoCommand):
    """Append a new connection."""

    def __init__(self, connections_tab: ConnectionsTab):
        """Initializes the connection append.

        Args:
            connections_tab (ConnectionsTab): reference to the connection widget
        """
        super().__init__('Append Connection')
        self.connections_tab = connections_tab

    def redo(self):
        """Appends a new event to the end of the list."""
        assert self.connections_tab.map_widget.main_gui.project is not None

        project = self.connections_tab.map_widget.main_gui.project
        datatype = self.connections_tab.map_widget.main_gui.project.config['pymap'][
            'header'
        ]['connections']['datatype']
        connections = self.connections_tab.map_widget.main_gui.get_connections()
        assert isinstance(connections, list)
        context = self.connections_tab.map_widget.main_gui.project.config['pymap'][
            'header'
        ]['connections']['connections_path'] + [len(connections)]
        parents = get_parents_by_path(
            self.connections_tab.map_widget.main_gui.header,
            self.connections_tab.map_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )

        connections.append(project.model[datatype](project, context, parents))
        self.connections_tab.map_widget.main_gui.set_number_of_connections(
            len(connections)
        )
        connections[-1] = unpack_connection(
            connections[-1], self.connections_tab.map_widget.main_gui.project
        )
        self.connections_tab.load_header()
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()

    def undo(self):
        """Removes the last event."""
        connections = self.connections_tab.map_widget.main_gui.get_connections()
        connections.pop()
        self.connections_tab.map_widget.main_gui.set_number_of_connections(
            len(connections)
        )
        self.connections_tab.load_header()
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()


class RemoveConnection(QUndoCommand):
    """Remove a connection."""

    def __init__(self, connections_tab: ConnectionsTab, connection_idx: int):
        """Initializes the connection removal.

        Args:
            connections_tab (ConnectionsTab): reference to the connection widget
            connection_idx (int): index of the connection
        """
        super().__init__('Remove Connection')
        self.connections_tab = connections_tab
        self.connection_idx = connection_idx
        self.connection = self.connections_tab.map_widget.main_gui.get_connections()[
            self.connection_idx
        ]

    def redo(self):
        """Removes the connection from the connections."""
        assert self.connections_tab.map_widget.main_gui.project is not None
        connections = self.connections_tab.map_widget.main_gui.get_connections()
        connections.pop(self.connection_idx)
        self.connections_tab.map_widget.main_gui.set_number_of_connections(
            len(connections)
        )
        self.connections_tab.load_header()
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()

    def undo(self):
        """Reinserts the connection."""
        assert self.connections_tab.map_widget.main_gui is not None
        assert self.connections_tab.map_widget.main_gui.header is not None
        assert self.connections_tab.map_widget.main_gui.project is not None
        connections = self.connections_tab.map_widget.main_gui.get_connections()
        assert isinstance(connections, list)
        connections.insert(self.connection_idx, self.connection)
        self.connections_tab.map_widget.main_gui.set_number_of_connections(
            len(connections)
        )
        connections[self.connection_idx] = unpack_connection(
            connections[self.connection_idx],
            self.connections_tab.map_widget.main_gui.project,
        )
        self.connections_tab.load_header()
        self.connections_tab.map_widget.update_blocks()
        self.connections_tab.map_widget.map_scene.update_connection_rectangles()
