"""Properties tree for connections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QWidget,
)

from agb.model.type import ModelContext, ModelValue
from pymap.gui.history import (
    ChangeConnectionProperty,
)
from pymap.gui.history.statement import (
    UndoRedoStatements,
)
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

if TYPE_CHECKING:
    from .connections import ConnectionsTab


class ConnectionProperties(PropertiesTree):
    """Tree to display connection properties."""

    def __init__(self, connections_tab: ConnectionsTab, parent: QWidget | None = None):
        """Initializes the event properties.

        Args:
            connections_tab (ConnectionsTab): The connection widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(
            'connections_tab',
            connections_tab.map_widget.main_gui,
            parent=parent,
        )
        self.connections_tab = connections_tab

    @property
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        assert self.connections_tab.map_widget.main_gui.project is not None, (
            'Project is None'
        )
        return self.connections_tab.map_widget.main_gui.project.config['pymap'][
            'header'
        ]['connections']['datatype']

    @property
    def model_value(self) -> ModelValue:
        """Returns the model value.

        Returns:
            Any: The model value.
        """
        if (
            not self.connections_tab.connection_loaded
            or self.connections_tab.idx_combobox.currentIndex() < 0
        ):
            raise ModelValueNotAvailableError()
        else:
            return self.connections_tab.map_widget.main_gui.get_connections()[
                self.connections_tab.idx_combobox.currentIndex()
            ]

    @property
    def model_context(self) -> ModelContext:
        """Returns the model context.

        Returns:
            ModelContext: The model context.
        """
        assert self.connections_tab.map_widget.main_gui.project is not None, (
            'Project is None'
        )
        return [self.connections_tab.idx_combobox.currentIndex()]

    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ) -> None:
        """Handles changes in the value.

        Args:
            statements_redo (UndoRedoStatements): Redo statements.
            statements_undo (UndoRedoStatements): Undo statements.
        """
        self.connections_tab.map_widget.undo_stack.push(
            ChangeConnectionProperty(
                self.connections_tab,
                self.connections_tab.idx_combobox.currentIndex(),
                self.connections_tab.mirror_offset.isChecked(),
                statements_redo,
                statements_undo,
            )
        )
