"""History actions for map headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from pymap.gui.history.statement import UndoRedoStatements

if TYPE_CHECKING:
    from pymap.gui.connection_widget import ConnectionWidget


class ChangeConnectionProperty(QUndoCommand):
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
        super().__init__()
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        self.mirror_offset = mirror_offset
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """Executes the redo statements."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list)
        root = connections[self.connection_idx]  # type: ignore # noqa: F841
        for statement in self.statements_redo:
            exec(statement)
        self.connection_widget.update_connection(
            self.connection_idx, self.mirror_offset
        )

    def undo(self):
        """Executes the redo statements."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list)
        root = connections[self.connection_idx]  # noqa: F841
        for statement in self.statements_undo:
            exec(statement)
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
        super().__init__()
        self.connection_widget = connection_widget

    def redo(self):
        """Appends a new event to the end of the list."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None

        project = self.connection_widget.main_gui.project
        datatype = self.connection_widget.main_gui.project.config['pymap']['header'][
            'connections'
        ]['datatype']
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list)
        context = self.connection_widget.main_gui.project.config['pymap']['header'][
            'connections'
        ]['connections_path'] + [len(connections)]
        parents = properties.get_parents_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        connections.append(project.model[datatype](project, context, parents))
        properties.set_member_by_path(
            self.connection_widget.main_gui.header,
            len(connections),
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_size_path'],
        )
        self.connection_widget.load_header()

    def undo(self):
        """Removes the last event."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None

        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        connections.pop()
        properties.set_member_by_path(
            self.connection_widget.main_gui.header,
            len(connections),
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_size_path'],
        )
        self.connection_widget.load_header()


class RemoveConnection(QUndoCommand):
    """Remove a connection."""

    def __init__(self, connection_widget: ConnectionWidget, connection_idx: int):
        """Initializes the connection removal.

        Args:
            connection_widget (ConnectionWidget): reference to the connection widget
            connection_idx (int): index of the connection
        """
        super().__init__()

        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        project = self.connection_widget.main_gui.project
        self.connection = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            project.config['pymap']['header']['connections']['connections_path'],
        )[self.connection_idx]

    def redo(self):
        """Removes the connection from the connections."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        connections.pop(self.connection_idx)
        properties.set_member_by_path(
            self.connection_widget.main_gui.header,
            len(connections),
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_size_path'],
        )
        self.connection_widget.load_header()

    def undo(self):
        """Reinserts the connection."""
        assert self.connection_widget.main_gui is not None
        assert self.connection_widget.main_gui.header is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list)
        connections.insert(self.connection_idx, self.connection)
        properties.set_member_by_path(
            self.connection_widget.main_gui.header,
            len(connections),
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_size_path'],
        )
        self.connection_widget.load_header()
