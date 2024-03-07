"""Properties tree for connections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepdiff import DeepDiff  # type: ignore
from pymap.gui import properties
from pymap.gui.history import (
    ChangeConnectionProperty,
    UndoRedoStatements,
)
from pymap.gui.types import Connection
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import (
    QHeaderView,
    QWidget,
)

from .child import ConnectionChildWidgetMixin, if_connection_loaded

if TYPE_CHECKING:
    from .connection_widget import ConnectionWidget


class ConnectionProperties(ParameterTree, ConnectionChildWidgetMixin):
    """Tree to display event properties."""

    def __init__(
        self, connection_widget: ConnectionWidget, parent: QWidget | None = None
    ):
        """Initializes the event properties.

        Args:
            connection_widget (ConnectionWidget): The connection widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        ConnectionChildWidgetMixin.__init__(self, connection_widget)
        self.connection_widget = connection_widget
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore

        self.root = None

    @if_connection_loaded
    def load_connection(self):
        """Loads the currently displayed connection."""
        self.clear()
        self.connection_widget

        if (
            self.connection_widget.main_gui.project is None
            or self.connection_widget.main_gui.header is None
            or self.connection_widget.idx_combobox.currentIndex() < 0
        ):
            self.root = None

        else:
            datatype = self.connection_widget.main_gui.project.config['pymap'][
                'header'
            ]['connections']['datatype']

            connections = properties.get_member_by_path(
                self.connection_widget.main_gui.header,
                self.connection_widget.main_gui.project.config['pymap']['header'][
                    'connections'
                ]['connections_path'],
            )
            assert isinstance(
                connections, list
            ), f'Expected list, got {type(connections)}'

            connection = connections[self.connection_widget.idx_combobox.currentIndex()]
            assert isinstance(
                connection, Connection
            ), f'Expected Connection, got {type(connection)}'

            parameter = properties.type_to_parameter(
                self.connection_widget.main_gui.project, datatype
            )
            self.root = parameter(
                '.',
                self.connection_widget.main_gui.project,
                datatype,
                connection,
                [self.connection_widget.idx_combobox.currentIndex()],
                None,
            )

            self.addParameters(self.root, showTop=False)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    @if_connection_loaded
    def update(self):
        """Updates all values in the tree according to the current connection."""
        assert self.root is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list), f'Expected list, got {type(connections)}'
        connection = connections[self.connection_widget.idx_combobox.currentIndex()]
        assert isinstance(
            connection, Connection
        ), f'Expected Connection, got {type(connection)}'

        self.root.blockSignals(True)  # type: ignore
        self.root.update(connection)
        self.root.blockSignals(False)  # type: ignore

    @if_connection_loaded
    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """When the tree changes values."""
        assert self.root is not None
        assert self.connection_widget.main_gui.project is not None
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header,
            self.connection_widget.main_gui.project.config['pymap']['header'][
                'connections'
            ]['connections_path'],
        )
        assert isinstance(connections, list), f'Expected list, got {type(connections)}'
        root = connections[self.connection_widget.idx_combobox.currentIndex()]
        assert isinstance(root, Connection), f'Expected Connection, got {type(root)}'
        diffs = DeepDiff(root, self.root.model_value)
        statements_redo: UndoRedoStatements = []
        statements_undo: UndoRedoStatements = []

        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:  # type: ignore
                    value_new = diffs[change][path]['new_value']  # type: ignore
                    value_old = diffs[change][path]['old_value']  # type: ignore
                    statements_redo.append(f"{path} = '{value_new}'")
                    statements_undo.append(f"{path} = '{value_old}'")
                    self.connection_widget.undo_stack.push(
                        ChangeConnectionProperty(
                            self.connection_widget,
                            self.connection_widget.idx_combobox.currentIndex(),
                            self.connection_widget.mirror_offset.isChecked(),
                            statements_redo,
                            statements_undo,
                        )
                    )
