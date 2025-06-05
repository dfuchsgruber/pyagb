"""History actions for map headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType
from pymap.gui import properties
from pymap.gui.history.statement import ChangeProperty, UndoRedoStatements

if TYPE_CHECKING:
    from pymap.gui.map.tabs.events.tab import EventsTab


class ChangeEventProperty(ChangeProperty):
    """Change a property of any event."""

    def __init__(
        self,
        events_tab: EventsTab,
        event_type: PymapEventConfigType,
        event_idx: int,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the event property change.

        Args:
            events_tab (EventsTab): reference to the event widget
            event_type (PymapEventConfigType): the event type
            event_idx (int): index of the event
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__(statements_redo, statements_undo, text='Change Event Property')
        self.events_tab = events_tab
        self.event_type = event_type
        self.event_idx = event_idx

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        return self.events_tab.map_widget.main_gui.get_event(
            self.event_type, self.event_idx
        )

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.events_tab.update_event(self.event_type, self.event_idx)

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.events_tab.update_event(self.event_type, self.event_idx)


class RemoveEvent(QUndoCommand):
    """Remove an event."""

    def __init__(
        self,
        events_tab: EventsTab,
        event_type: PymapEventConfigType,
        event_idx: int,
    ):
        """Initializes the event removal.

        Args:
            events_tab (EventsTab): reference to the event widget
            event_type (PymapEventConfigType): the event type
            event_idx (int): index of the event
        """
        super().__init__('Remove Event')
        self.events_tab = events_tab
        self.event_type = event_type
        self.event_idx = event_idx
        self.event = self.events_tab.map_widget.main_gui.get_event(
            self.event_type, self.event_idx
        )

    def redo(self):
        """Removes the event from the events."""
        events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
        assert self.events_tab.map_widget.main_gui is not None
        assert self.events_tab.map_widget.main_gui.header is not None
        events.pop(self.event_idx)
        self.events_tab.map_widget.main_gui.set_number_of_events(
            self.event_type, len(events)
        )
        self.events_tab.map_widget.map_scene.remove_event_image(
            self.event_type, self.event_idx
        )
        self.events_tab.load_header()

    def undo(self):
        """Reinserts the event."""
        events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
        events.insert(self.event_idx, self.event)
        assert self.events_tab.map_widget.main_gui is not None
        assert self.events_tab.map_widget.main_gui.header is not None
        self.events_tab.map_widget.main_gui.set_number_of_events(
            self.event_type, len(events)
        )
        self.events_tab.map_widget.map_scene.insert_event_image(
            self.event_type, self.event_idx, self.event
        )
        self.events_tab.load_header()


class AppendEvent(QUndoCommand):
    """Append a new event."""

    def __init__(
        self,
        events_tab: EventsTab,
        event_type: PymapEventConfigType,
        event: ModelValue | None = None,
    ):
        """Initializes the event appending.

        Args:
            events_tab (EventsTab): reference to the event widget
            event_type (PymapEventConfigType): the event type
            event (ModelValue | None): optional event to append.
                If None a new one is created.
        """
        super().__init__(
            'Append Event',
        )
        self.events_tab = events_tab
        self.event_type = event_type
        if event is None:
            assert self.events_tab.map_widget.main_gui is not None
            assert self.events_tab.map_widget.main_gui.project is not None
            project = self.events_tab.map_widget.main_gui.project
            datatype = self.event_type['datatype']
            events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
            context = list(self.event_type['events_path']) + [len(events)]
            assert self.events_tab.map_widget.main_gui.header is not None
            parents = properties.get_parents_by_path(
                self.events_tab.map_widget.main_gui.header, context
            )
            event = project.model[datatype](project, context, parents)
        self.event = event

    def redo(self):
        """Appends a new event to the end of the list."""
        events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
        events.append(self.event)
        self.events_tab.map_widget.main_gui.set_number_of_events(
            self.event_type, len(events)
        )
        self.events_tab.map_widget.map_scene.insert_event_image(
            self.event_type, len(events) - 1, events[-1]
        )
        self.events_tab.load_header()

    def undo(self):
        """Removes the last event."""
        assert self.events_tab.map_widget.main_gui is not None
        assert self.events_tab.map_widget.main_gui.header is not None
        events = self.events_tab.map_widget.main_gui.get_events(self.event_type)
        assert isinstance(events, list), f'Expected list, got {type(events)}'
        events.pop()
        self.events_tab.map_widget.main_gui.set_number_of_events(
            self.event_type, len(events)
        )
        self.events_tab.map_widget.map_scene.remove_event_image(
            self.event_type, len(events)
        )
        self.events_tab.load_header()
