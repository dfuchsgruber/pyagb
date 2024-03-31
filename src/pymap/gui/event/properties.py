"""Properties tree for events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelValue
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import (
    QHeaderView,
    QWidget,
)
from typing_extensions import ParamSpec

from pymap.gui import properties
from pymap.gui.history import (
    model_value_difference_to_undo_redo_statements,
)

from ..history import ChangeEventProperty

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tab import EventTab


class EventProperties(ParameterTree):
    """Tree to display event properties."""

    def __init__(self, event_tab: EventTab, parent: QWidget | None = None):
        """Initializes the event properties.

        Args:
            event_tab (EventTab): The event tab.
            parent (QWidget | None, optional): Parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        self.event_tab = event_tab
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.header().restoreState(  # type: ignore
            self.event_tab.event_widget.main_gui.settings.value(
                f'event_widget_{self.event_tab.event_type["name"]}/header_state',
                b'',
                type=bytes,
            )
        )
        self.header().sectionResized.connect(  # type: ignore
            lambda: self.event_tab.event_widget.main_gui.settings.setValue(  # type: ignore
                f'event_widget_{self.event_tab.event_type["name"]}/header_state',
                self.header().saveState(),  # type: ignore
            )
        )
        self.root = None

    def _get_current_event(self) -> ModelValue:
        """Returns the currently displayed event.

        Returns:
            ModelValue: The currently displayed event.
        """
        return self.event_tab.event_widget.main_gui.get_event(
            self.event_tab.event_type, self.event_tab.idx_combobox.currentIndex()
        )

    def load_event(self):
        """Loads the currently displayed event."""
        self.clear()
        if (
            self.event_tab.event_widget.main_gui.project is None
            or self.event_tab.event_widget.main_gui.header is None
            or self.event_tab.idx_combobox.currentIndex() < 0
        ):
            self.root = None
        else:
            datatype = self.event_tab.event_type['datatype']
            event = self._get_current_event()
            self.root = properties.type_to_parameter(
                self.event_tab.event_widget.main_gui.project, datatype
            )(
                '.',
                self.event_tab.event_widget.main_gui.project,
                datatype,
                event,
                self.event_tab.event_type['events_path']
                + [self.event_tab.idx_combobox.currentIndex()],
                None,
            )
            self.addParameters(self.root, showTop=False)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def update(self):
        """Updates all values in the tree according to the current event."""
        event = self._get_current_event()

        assert self.root is not None, 'Root is None'
        self.root.blockSignals(True)  # type: ignore
        self.root.update(event)
        self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Signal handler for when the tree changes.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        assert self.root is not None, 'Root is None'
        root = self._get_current_event()
        self.event_tab.event_widget.undo_stack.push(
            ChangeEventProperty(
                self.event_tab.event_widget,
                self.event_tab.event_type,
                self.event_tab.idx_combobox.currentIndex(),
                *model_value_difference_to_undo_redo_statements(
                    root, self.root.model_value
                ),
            )
        )
