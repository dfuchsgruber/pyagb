"""Properties tree for connections."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelContext, ModelValue
from PySide6.QtWidgets import (
    QWidget,
)

from pymap.gui.history import (
    ChangeConnectionProperty,
)
from pymap.gui.history.statement import (
    UndoRedoStatements,
)
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

if TYPE_CHECKING:
    from .connection_widget import ConnectionWidget


class ConnectionProperties(PropertiesTree):
    """Tree to display connection properties."""

    def __init__(
        self, connection_widget: ConnectionWidget, parent: QWidget | None = None
    ):
        """Initializes the event properties.

        Args:
            connection_widget (ConnectionWidget): The connection widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(
            'connection_widget',
            connection_widget.main_gui,
            parent=parent,
        )
        self.connection_widget = connection_widget

    @property
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        assert self.connection_widget.main_gui.project is not None, 'Project is None'
        return self.connection_widget.main_gui.project.config['pymap']['header'][
            'connections'
        ]['datatype']

    @property
    def model_value(self) -> ModelValue:
        """Returns the model value.

        Returns:
            Any: The model value.
        """
        if (
            not self.connection_widget.connection_loaded
            or self.connection_widget.idx_combobox.currentIndex() < 0
        ):
            raise ModelValueNotAvailableError()
        else:
            return self.connection_widget.main_gui.get_connections()[
                self.connection_widget.idx_combobox.currentIndex()
            ]

    @property
    def model_context(self) -> ModelContext:
        """Returns the model context.

        Returns:
            ModelContext: The model context.
        """
        assert self.connection_widget.main_gui.project is not None, 'Project is None'
        return [self.connection_widget.idx_combobox.currentIndex()]

    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ) -> None:
        """Handles changes in the value.

        Args:
            statements_redo (UndoRedoStatements): Redo statements.
            statements_undo (UndoRedoStatements): Undo statements.
        """
        self.connection_widget.undo_stack.push(
            ChangeConnectionProperty(
                self.connection_widget,
                self.connection_widget.idx_combobox.currentIndex(),
                self.connection_widget.mirror_offset.isChecked(),
                statements_redo,
                statements_undo,
            )
        )
