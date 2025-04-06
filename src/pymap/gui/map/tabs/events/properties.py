"""Properties tree for events."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from PySide6.QtWidgets import (
    QWidget,
)

from agb.model.type import ModelContext, ModelValue
from pymap.gui.history.statement import UndoRedoStatements
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

from ....history import ChangeEventProperty

if TYPE_CHECKING:
    from .tab import EventTab


class EventProperties(PropertiesTree):
    """Tree to display event properties."""

    def __init__(self, event_tab: EventTab, parent: QWidget | None = None):
        """Initializes the event properties.

        Args:
            event_tab (EventTab): The event tab.
            parent (QWidget | None, optional): Parent. Defaults to None.
        """
        self.event_tab = event_tab
        super().__init__(
            f'event_{self.event_tab.event_type["name"]}',
            self.event_tab.events_tab.map_widget.main_gui,
            parent=parent,
        )

    @property
    @override
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        return self.event_tab.event_type['datatype']

    @property
    @override
    def model_value(self) -> ModelValue:
        """Returns the model value.

        Returns:
            ModelValue: The model value.

        Raises:
            ModelValueNotAvailableError: If the model value is not available.
        """
        if (
            self.event_tab.events_tab.map_widget.main_gui.project is None
            or self.event_tab.events_tab.map_widget.main_gui.header is None
            or self.event_tab.idx_combobox.currentIndex() < 0
        ):
            raise ModelValueNotAvailableError()
        return self.event_tab.events_tab.map_widget.main_gui.get_event(
            self.event_tab.event_type, self.event_tab.idx_combobox.currentIndex()
        )

    @property
    @override
    def model_context(self) -> ModelContext:
        """Returns the model context.

        Returns:
            list[ModelValue]: The model context.
        """
        return self.event_tab.event_type['events_path'] + [
            self.event_tab.idx_combobox.currentIndex()
        ]

    @override
    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ):
        """Signal handler for when the value changes.

        Args:
            statements_redo (UndoRedoStatements): Statements for redo.
            statements_undo (UndoRedoStatements): Statements for undo.
        """
        self.event_tab.events_tab.map_widget.undo_stack.push(
            ChangeEventProperty(
                self.event_tab.events_tab,
                self.event_tab.event_type,
                self.event_tab.idx_combobox.currentIndex(),
                statements_redo,
                statements_undo,
            )
        )
