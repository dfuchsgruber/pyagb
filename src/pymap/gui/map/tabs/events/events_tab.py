"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Protocol, cast

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QTabWidget,
    QWidget,
)

from agb.model.type import ModelValue
from pymap.configuration import EventTemplateType, PymapEventConfigType
from pymap.gui import properties
from pymap.gui.history.event import AppendEvent, ChangeEventProperty, RemoveEvent
from pymap.gui.history.statement import (
    model_value_difference_to_undo_redo_statements,
    path_to_statement,
)
from pymap.gui.map.view.layers.layer import VisibleLayer
from pymap.gui.properties.utils import get_member_by_path, set_member_by_path

from ..tab import MapWidgetTab
from .tab import EventTab

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class EventsTab(MapWidgetTab):
    """Tab for the events on the map."""

    class TemplateBasedContextMenuAction(Protocol):
        """Protocol for a context menu action that is based on a template."""

        def __call__(
            self,
            template: EventTemplateType = [],
        ) -> ModelValue:
            """Call the action with the given parameters."""
            ...

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

        # Layout is similar to the map widget
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._tab_changed)
        self.tabs = {}
        layout.addWidget(self.tab_widget, 1, 1, 1, 1)

        # Recording mouse events
        self.last_dragged_position: tuple[int, int] | None = None
        self.dragged_event = None
        self.is_dragging = False

    def _tab_changed(self, index: int) -> None:
        """Event handler for changing the tab.

        Args:
            index (int): The index of the new tab.
        """
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
        ):
            return
        tab: EventTab = cast(EventTab, self.tab_widget.currentWidget())
        tab.select_event()

    def load_project(self):
        """Loads the project."""
        assert self.map_widget.main_gui.project is not None, 'Project is None'

        self.tab_widget.clear()
        self.tabs: dict[str, EventTab] = {}

        for name, event_type in self.map_widget.main_gui.project.config['pymap'][
            'header'
        ]['events'].items():
            self.tabs[event_type['datatype']] = EventTab(self, name, event_type)
            self.tab_widget.addTab(self.tabs[event_type['datatype']], name)

    def load_header(self):
        """Loads the tab."""
        self.load_events()

    def load_events(self):
        """Loads all event images."""
        for datatype in self.tabs:
            self.tabs[datatype].load_events()
        self._tab_changed(self.tab_widget.currentIndex())

    def update_event(self, event_type: PymapEventConfigType, event_idx: int):
        """Updates an event."""
        assert event_idx >= 0, 'Event index is negative'
        self.map_widget.map_scene_view.events.update_event_image(event_type, event_idx)
        tab = self.tabs[event_type['datatype']]
        if tab.idx_combobox.currentIndex() == event_idx:
            # If the event is selected in its tab, it's properties tree should
            # be updated
            tab.event_properties.update()
            if self.tab_widget.currentWidget() == tab:
                # If the tab is selected and the event is selected, update the
                # rectangle in the map scene
                # TODO: this only needs to be done if the event moves its coordinates
                self.map_widget.map_scene_view.selected_event.update_selected_event_image(
                    event_type, event_idx
                )

    @property
    def visible_layers(self) -> VisibleLayer:
        """Get the visible layers."""
        return (
            VisibleLayer.BLOCKS
            | VisibleLayer.EVENTS
            | VisibleLayer.SELECTED_EVENT
            | VisibleLayer.BORDER_EFFECT
            | VisibleLayer.GRID
        )

    def _get_event_by_padded_position(
        self,
        x: int,
        y: int,
    ) -> tuple[PymapEventConfigType, int] | None:
        """Gets the event associated with a mouse position."""
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
        ):
            return None
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        padded_x, padded_y = self.map_widget.main_gui.get_border_padding()

        if x - padded_x in range(map_width) and y - padded_y in range(map_height):
            # Check if there is any event that can be picked up
            for _, event_type in self.map_widget.main_gui.project.config['pymap'][
                'header'
            ]['events'].items():
                events = self.map_widget.main_gui.get_events(event_type)
                for event_idx, event in enumerate(events):
                    event_x, event_y = self.map_widget.map_scene_view.pad_coordinates(
                        get_member_by_path(event, event_type['x_path']),
                        get_member_by_path(event, event_type['y_path']),
                        padded_x,
                        padded_y,
                    )

                    if x == event_x and event_y == y:
                        return event_type, event_idx
        return None

    def _select_event_at_padded_position(
        self,
        x: int,
        y: int,
    ) -> tuple[PymapEventConfigType, int] | None:
        """Selects an event at the given position."""
        target = self._get_event_by_padded_position(x, y)
        if target is not None:
            event_type, event_idx = target

            # Change to the tab and select the event
            tab = self.tabs[event_type['datatype']]
            if self.tab_widget.currentWidget() != tab:
                self.tab_widget.setCurrentWidget(tab)
            if tab.idx_combobox.currentIndex() != event_idx:
                tab.idx_combobox.setCurrentIndex(event_idx)

            # Update the mouse events
            self.last_dragged_position = x, y
            self.dragged_event = event_type, event_idx
            self.is_dragging = False
        else:
            # If no event is selected, set the dragged event to None
            self.last_dragged_position = None
            self.dragged_event = None
            self.is_dragging = False
        return target

    def map_scene_mouse_pressed(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._select_event_at_padded_position(x, y)
        elif event.button() == Qt.MouseButton.RightButton:
            self._select_event_at_padded_position(x, y)
            self.create_map_scene_context_menu_at(event, x, y)

    def create_map_scene_context_menu_at(self, event: QMouseEvent, x: int, y: int):
        """Creates a context menu at the given position.

        Args:
            event (QMouseEvent): The event that triggered the context menu.
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        # Create a context menu
        context_menu = QtWidgets.QMenu()

        # Get the delete action if there is a target event
        target = self._get_event_by_padded_position(x, y)
        if target is not None:
            event_type, event_idx = target
            # Add a delete action to the context menu
            action = context_menu.addAction('Delete')  # type: ignore
            action.triggered.connect(
                partial(
                    self.map_widget.undo_stack.push,
                    RemoveEvent(self, event_type, event_idx),
                )
            )
            context_menu.addSeparator()
            submenu = context_menu.addMenu('Set')
            self._context_menu_add_template_actions(
                submenu,
                event_type,
                partial(self._set_default_event, event_idx, event_type),
            )
            context_menu.addSeparator()

        assert self.map_widget.main_gui.project is not None, 'Project is None'
        assert self.map_widget.main_gui.header is not None, 'Header is None'
        for name, event_type in self.map_widget.main_gui.project.config['pymap'][
            'header'
        ]['events'].items():
            # Create a submenu for each event type
            submenu = context_menu.addMenu(f'Add {name}')
            self._context_menu_add_template_actions(
                submenu,
                event_type,
                partial(self._add_default_event_at, event_type, x, y),
            )
        # Execute the context menu at the given position
        self.context_menu = context_menu
        self.context_menu.popup(event.screenPos().toPoint())

    def _context_menu_add_template_actions(
        self,
        context_menu: QtWidgets.QMenu,
        event_type: PymapEventConfigType,
        action_function: TemplateBasedContextMenuAction,
    ):
        """Adds actions to the context menu for each event type based on the templates.

        Args:
            context_menu (QtWidgets.QMenu): The context menu to add actions to.
            event_type (PymapEventConfigType): The event type to add actions for.
            action_function (TemplateBasedContextMenuAction): The function to call when
                an action is triggered.
        """
        assert self.map_widget.main_gui.project is not None, 'Project is None'
        assert self.map_widget.main_gui.header is not None, 'Header is None'
        # Add a 'default' action to the submenu
        action = context_menu.addAction('Default')  # type: ignore
        action.triggered.connect(
            partial(
                action_function,
            )
        )
        templates = event_type['templates']
        if templates:
            context_menu.addSeparator()
            # Add all templates as actions to the submenu
            for name, template in templates.items():
                action = context_menu.addAction(name)  # type: ignore
                action.triggered.connect(
                    partial(
                        action_function,
                        template=template,
                    )
                )

    def get_default_event(
        self,
        event_type: PymapEventConfigType,
        x: ModelValue | None = None,
        y: ModelValue | None = None,
    ) -> ModelValue:
        """Gets a default event for the given event type.

        Args:
            event_type (PymapEventConfigType): The event type to get the default event
                for.
            x (ModelValue | None): The x coordinate of the event. If None, the default
                value is used.
            y (ModelValue | None): The y coordinate of the event. If None, the default
                value is used.

        Returns:
            ModelValue: The default event.
        """
        assert self.map_widget.main_gui is not None
        assert self.map_widget.main_gui.project is not None
        project = self.map_widget.main_gui.project
        datatype = event_type['datatype']
        events = self.map_widget.main_gui.get_events(event_type)
        context = list(event_type['events_path']) + [len(events)]
        assert self.map_widget.main_gui.header is not None
        parents = properties.get_parents_by_path(
            self.map_widget.main_gui.header, context
        )
        event = project.model[datatype](project, context, parents)
        if x is not None:
            set_member_by_path(event, x, event_type['x_path'])
        if y is not None:
            set_member_by_path(event, y, event_type['y_path'])
        return event

    def get_event_from_template(
        self,
        event_type: PymapEventConfigType,
        template: EventTemplateType = [],
        x: ModelValue | None = None,
        y: ModelValue | None = None,
    ) -> ModelValue:
        """Gets an event from a template.

        Args:
            event_type (PymapEventConfigType): The event type to get the event for.
            template (EventTemplateType): The template to use for the event.
            x (ModelValue | None): The x coordinate of the event. If None, the default
                value is used.
            y (ModelValue | None): The y coordinate of the event. If None, the default
                value is used.

        Returns:
            ModelValue: The event.
        """
        event = self.get_default_event(event_type, x=x, y=y)
        for path, value in template:
            set_member_by_path(event, value, path)
        return event

    def _add_default_event_at(
        self,
        event_type: PymapEventConfigType,
        x: int,
        y: int,
        template: EventTemplateType = [],
    ) -> ModelValue:
        """Adds a new default event at the given position."""
        padded_x, padded_y = self.map_widget.main_gui.get_border_padding()
        event = self.get_event_from_template(
            event_type, template, x=x - padded_x, y=y - padded_y
        )
        item = AppendEvent(self, event_type, event)
        event_idx = self.map_widget.main_gui.get_num_events(event_type)
        self.map_widget.undo_stack.push(item)

        tab = self.tabs[event_type['datatype']]
        if self.tab_widget.currentWidget() != tab:
            self.tab_widget.setCurrentWidget(tab)
        if tab.idx_combobox.currentIndex() != event_idx:
            tab.idx_combobox.setCurrentIndex(event_idx)

        return item.event

    def _set_default_event(
        self,
        event_idx: int,
        event_type: PymapEventConfigType,
        template: EventTemplateType = [],
    ) -> ModelValue:
        """Sets an event at a given index to a default value from a template."""
        event = self.get_event_from_template(
            event_type,
            template,
            x=get_member_by_path(
                self.map_widget.main_gui.get_event(event_type, event_idx),
                event_type['x_path'],
            ),
            y=get_member_by_path(
                self.map_widget.main_gui.get_event(event_type, event_idx),
                event_type['y_path'],
            ),
        )
        statements_redo, statements_undo = (
            model_value_difference_to_undo_redo_statements(
                self.map_widget.main_gui.get_event(event_type, event_idx),
                event,
            )
        )
        item = ChangeEventProperty(
            self, event_type, event_idx, statements_redo, statements_undo
        )
        self.map_widget.undo_stack.push(item)
        tab = self.tabs[event_type['datatype']]
        if self.tab_widget.currentWidget() != tab:
            self.tab_widget.setCurrentWidget(tab)
        if tab.idx_combobox.currentIndex() != event_idx:
            tab.idx_combobox.setCurrentIndex(event_idx)

        return event

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_event_at_padded_position(x, y)

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_event_at_padded_position(x, y)

    def map_scene_mouse_moved(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        if self.last_dragged_position is not None and self.last_dragged_position != (
            x,
            y,
        ):
            assert self.dragged_event is not None, 'Dragged event is None'
            event_type, event_idx = self.dragged_event
            padded_x, padded_y = self.map_widget.main_gui.get_border_padding()

            # Start the dragging macro if not started already
            if not self.is_dragging:
                self.is_dragging = True
                self.map_widget.undo_stack.beginMacro(
                    f'Drag event {event_type["datatype"]} {event_idx}'
                )

            # Drag the current event to this position
            redo_statement_x, undo_statement_x = path_to_statement(
                event_type['x_path'],
                self.last_dragged_position[0] - padded_x,
                x - padded_x,
            )
            redo_statement_y, undo_statement_y = path_to_statement(
                event_type['y_path'],
                self.last_dragged_position[1] - padded_y,
                y - padded_y,
            )
            self.map_widget.undo_stack.push(
                ChangeEventProperty(
                    self,
                    event_type,
                    event_idx,
                    [redo_statement_x, redo_statement_y],
                    [undo_statement_x, undo_statement_y],
                )
            )
            self.last_dragged_position = x, y

    def map_scene_mouse_released(self, event: QMouseEvent, x: int, y: int) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if self.is_dragging:
            self.is_dragging = False
            self.map_widget.undo_stack.endMacro()
        self.last_dragged_position = None
        self.dragged_event = None

    def map_scene_mouse_double_clicked(
        self, event: QMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for double clicking the mouse."""
        if not self.map_widget.header_loaded:
            return
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        target = self._get_event_by_padded_position(x, y)
        if target is None:
            return
        event_type, event_idx = target
        tab = self.tabs[event_type['datatype']]
        if event_type.get('goto_header_button_button_enabled', False):
            tab.goto_header(event_idx)

    def shift_events(self, x: int, y: int):
        """Shifts the events of the current map header."""
        if not self.map_widget.main_gui.header_loaded:
            return
        self.map_widget.undo_stack.beginMacro('ShiftEvents')
        assert self.map_widget.main_gui.project is not None, 'Project is not loaded'
        for _, event_type in self.map_widget.main_gui.project.config['pymap']['header'][
            'events'
        ].items():
            for event_idx in range(self.map_widget.main_gui.get_num_events(event_type)):
                event = self.map_widget.main_gui.get_event(event_type, event_idx)

                x_old = eval(str(get_member_by_path(event, event_type['x_path'])))
                y_old = eval(str(get_member_by_path(event, event_type['y_path'])))
                redo_statement_x, undo_statement_x = path_to_statement(
                    event_type['x_path'], x_old, x_old + x
                )
                redo_statement_y, undo_statement_y = path_to_statement(
                    event_type['y_path'], y_old, y_old + y
                )
                self.map_widget.undo_stack.push(
                    ChangeEventProperty(
                        self,
                        event_type,
                        event_idx,
                        [redo_statement_x, redo_statement_y],
                        [undo_statement_x, undo_statement_y],
                    )
                )
        self.map_widget.undo_stack.endMacro()
