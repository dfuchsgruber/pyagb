"""History actions for map headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelValue
from pymap.gui.history.statement import ChangeProperty, UndoRedoStatements
from pymap.gui.properties import get_parents_by_path

if TYPE_CHECKING:
    from pymap.gui.connection import ConnectionWidget


class ChangeConnectionProperty(ChangeProperty):
    """Change a property of any vent."""

    def __init__(
        self,
        connection_widget: ConnectionWidget,
        connection_idx: int,
        mirror_offset: bool,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the event property change.

        Args:
            connection_widget (ConnectionWidget): reference to the connection widget
            connection_idx (int): index of the connection
            mirror_offset (bool): whether the connection is mirrored
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__(
            statements_redo, statements_undo, text='Change Connection Property'
        )
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        self.mirror_offset = mirror_offset

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        connections = self.connection_widget.main_gui.get_connections()
        return connections[self.connection_idx]

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.connection_widget.update_connection(
            self.connection_idx, self.mirror_offset
        )

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.connection_widget.update_connection(
            self.connection_idx, self.mirror_offset
        )


class AppendConnection(QUndoCommand):
    """Append a new connection."""

    def __init__(self, connection_widget: ConnectionWidget):
        """Initializes the connection append.

        Args:
            connection_widget (ConnectionWidget): reference to the connection widget
        """
        super().__init__('Append Connection')
        self.connection_widget = connection_widget

    def redo(self):
        """Appends a new event to the end of the list."""
        assert self.connection_widget.main_gui.project is not None

        project = self.connection_widget.main_gui.project
        datatype = self.connection_widget.main_gui.project.config['pymap']['header'][
            'connections'
        ]['datatype']
        connections = self.connection_widget.main_gui.get_connections()
        assert isinstance(connections, list)
        context = self.connection_widget.main_gui.project.config['pymap']['header'][
            'connections'
        ]['connections_path'] + [len(connections)]
        parents = get_parents_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )

        connections.append(project.model[datatype](project, context, parents))
        self.connection_widget.main_gui.set_number_of_connections(len(connections))
        self.connection_widget.load_header()

    def undo(self):
        """Removes the last event."""
        connections = self.connection_widget.main_gui.get_connections()
        connections.pop()
        self.connection_widget.main_gui.set_number_of_connections(len(connections))
        self.connection_widget.load_header()


class RemoveConnection(QUndoCommand):
    """Remove a connection."""

    def __init__(self, connection_widget: ConnectionWidget, connection_idx: int):
        """Initializes the connection removal.

        Args:
            connection_widget (ConnectionWidget): reference to the connection widget
            connection_idx (int): index of the connection
        """
        super().__init__('Remove Connection')
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        self.connection = self.connection_widget.main_gui.get_connections()[
            self.connection_idx
        ]

    def redo(self):
        """Removes the connection from the connections."""
        assert self.connection_widget.main_gui.project is not None
        connections = self.connection_widget.main_gui.get_connections()
        connections.pop(self.connection_idx)
        self.connection_widget.main_gui.set_number_of_connections(len(connections))
        self.connection_widget.load_header()

    def undo(self):
        """Reinserts the connection."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        connections = self.connection_widget.main_gui.get_connections()
        assert isinstance(connections, list)
        connections.insert(self.connection_idx, self.connection)
        self.connection_widget.main_gui.set_number_of_connections(len(connections))
        self.connection_widget.load_header()
