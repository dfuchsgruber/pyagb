"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QSplitter,
    QTabWidget,
    QWidget,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui.history.event import ChangeEventProperty
from pymap.gui.history.statement import path_to_statement
from pymap.gui.map_scene import MapScene
from pymap.gui.properties.utils import get_member_by_path

from ..tab import MapWidgetTab
from .tab import EventTab

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class EventsTab(MapWidgetTab):
    """Tab for the events on the map."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

        # Layout is similar to the map widget
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        splitter = QSplitter()
        splitter.restoreState(
            self.map_widget.main_gui.settings.value(
                'EventWidget/splitterState', b'', type=bytes
            )  # type: ignore
        )
        splitter.splitterMoved.connect(
            lambda: self.map_widget.main_gui.settings.setValue(
                'EventWidget/splitterState', splitter.saveState()
            )
        )
        layout.addWidget(splitter, 1, 1, 1, 1)

        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label, 2, 1, 1, 2)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._tab_changed)
        self.tabs = {}
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([4 * 10**6, 10**6])  # Ugly as hell hack to take large values

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

        for event_type in self.map_widget.main_gui.project.config['pymap']['header'][
            'events'
        ]:
            self.tabs[event_type['datatype']] = EventTab(self, event_type)
            self.tab_widget.addTab(
                self.tabs[event_type['datatype']], event_type['name']
            )

        # Load backend for event_to_image

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
        self.map_widget.map_scene.update_event_image(event_type, event_idx)
        tab = self.tabs[event_type['datatype']]
        if tab.idx_combobox.currentIndex() == event_idx:
            # If the event is selected in its tab, it's properties tree should
            # be updated
            tab.event_properties.update()
            if self.tab_widget.currentWidget() == tab:
                # If the tab is selected and the event is selected, update the
                # rectangle in the map scene
                # TODO: this only needs to be done if the event moves its coordinates
                self.map_widget.map_scene.update_selected_event_image(
                    event_type, event_idx
                )

    @property
    def visible_layers(self) -> MapScene.VisibleLayer:
        """Get the visible layers."""
        return (
            MapScene.VisibleLayer.BLOCKS
            | MapScene.VisibleLayer.EVENTS
            | MapScene.VisibleLayer.BORDER_EFFECT
            | MapScene.VisibleLayer.CONNECTIONS
            | MapScene.VisibleLayer.SELECTED_EVENT
        )

    def _get_event_by_padded_position(
        self,
        x: int,
        y: int,
    ) -> tuple[PymapEventConfigType, int] | None:
        """Finds an event associated with a mouse event by the position of the mouse event."""
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
        ):
            return None
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        padded_x, padded_y = self.map_widget.main_gui.get_border_padding()

        if x - padded_x in range(map_width) and y - padded_y in range(map_height):
            # Check if there is any event that can be picked up
            for event_type in self.map_widget.main_gui.project.config['pymap'][
                'header'
            ]['events']:
                events = self.map_widget.main_gui.get_events(event_type)
                for event_idx, event in enumerate(events):
                    event_x, event_y = self.map_widget.map_scene.pad_coordinates(
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
    ):
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

    def map_scene_mouse_pressed(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if not self.map_widget.header_loaded:
            return
        if not event.button() == Qt.MouseButton.LeftButton:
            return
        self._select_event_at_padded_position(x, y)

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

    def map_scene_mouse_moved(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
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
                    f'Drag event {event_type["name"]} {event_idx}'
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

    def map_scene_mouse_released(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
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
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
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
