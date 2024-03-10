"""Widget for the event map scene."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui.history import path_to_statement
from pymap.gui.properties import get_member_by_path

from ..history import ChangeEventProperty
from .child import EventChildWidgetMixin, if_header_loaded
from .tab import EventTab, pad_coordinates

if TYPE_CHECKING:
    from .event_widget import EventWidget


class MapScene(QGraphicsScene, EventChildWidgetMixin):
    """Scene for the map view."""

    def __init__(self, event_widget: EventWidget, parent: QWidget | None = None):
        """Initializes the map scene.

        Args:
            event_widget (EventWidget): The event widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        EventChildWidgetMixin.__init__(self, event_widget)

        self.event_groups = defaultdict(list)  # Items for each event type
        # Store the position where a drag happend recently so there are not
        # multiple draw events per block
        self.last_drag = None
        # Tuple to store the dragged event type and index
        self.dragged_event = None
        # Indicate if the event has at least moved one block, i.e.
        # if a macro is currently active
        self.drag_started = False

    def update_selection(self):
        """Updates the selection rectangle to match the currently selected item."""
        if self.selection_rect is not None:
            self.removeItem(self.selection_rect)
            self.selection_rect = None
        if (
            self.event_widget.main_gui.project is None
            or self.event_widget.main_gui.header is None
        ):
            return
        tab: EventTab | None = self.event_widget.tab_widget.currentWidget()  # type: ignore
        if tab is None:
            return
        idx = tab.idx_combobox.currentIndex()

        event = self.event_widget.main_gui.get_event(tab.event_type, idx)

        padded_x, padded_y = self.event_widget.main_gui.get_border_padding()

        x, y = pad_coordinates(
            get_member_by_path(event, tab.event_type['x_path']),
            get_member_by_path(event, tab.event_type['y_path']),
            padded_x,
            padded_y,
        )
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        self.selection_rect = self.addRect(
            x, y, 16, 16, pen=QPen(color, 2.0), brush=QBrush(0)
        )

    def clear(self):
        """Clears the map scene."""
        super().clear()
        self.event_groups: dict[str, list[QGraphicsItemGroup]] = defaultdict(list)
        self.selection_rect = None

    @if_header_loaded
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for hover events on the map image."""
        map_width, map_height = self.event_widget.main_gui.get_map_dimensions()
        padded_x, padded_y = self.event_widget.main_gui.get_border_padding()

        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x - padded_x in range(map_width) and y - padded_y in range(map_height):
            self.event_widget.info_label.setText(
                f'x : {hex(x - padded_x)}, y : {hex(y - padded_y)}'
            )
            if self.last_drag is not None and self.last_drag != (x, y):
                # Start the dragging macro if not started already
                if not self.drag_started:
                    self.drag_started = True
                    self.event_widget.undo_stack.beginMacro('Drag event')

                # Drag the current event to this position
                assert self.dragged_event is not None, 'Dragged event is None'
                event_type, event_idx = self.dragged_event
                # Assemble undo and redo instructions for changing the coordinates
                redo_statement_x, undo_statement_x = path_to_statement(
                    event_type['x_path'], self.last_drag[0] - padded_x, x - padded_x
                )
                redo_statement_y, undo_statement_y = path_to_statement(
                    event_type['y_path'], self.last_drag[1] - padded_y, y - padded_y
                )
                self.event_widget.undo_stack.push(
                    ChangeEventProperty(
                        self.event_widget,
                        event_type,
                        event_idx,
                        [redo_statement_x, redo_statement_y],
                        [undo_statement_x, undo_statement_y],
                    )
                )
                self.last_drag = x, y
        else:
            self.event_widget.info_label.setText('')

    def _get_event_by_event_position(
        self, mouse_event: QGraphicsSceneMouseEvent
    ) -> tuple[PymapEventConfigType, int] | None:
        """Finds an event associated with a mouse event by the position of the mouse event."""
        if (
            self.event_widget.main_gui.project is None
            or self.event_widget.main_gui.header is None
        ):
            return None
        map_width, map_height = self.event_widget.main_gui.get_map_dimensions()
        padded_x, padded_y = self.event_widget.main_gui.get_border_padding()

        pos = mouse_event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if (
            x - padded_x in range(map_width)
            and y - padded_y in range(map_height)
            and mouse_event.button() == Qt.MouseButton.LeftButton
        ):
            # Check if there is any event that can be picked up
            for event_type in self.event_widget.main_gui.project.config['pymap'][
                'header'
            ]['events']:
                events = get_member_by_path(
                    self.event_widget.main_gui.header, event_type['events_path']
                )
                assert isinstance(events, list), f'Expected list, got {type(events)}'
                for event_idx, event in enumerate(events):
                    event_x, event_y = pad_coordinates(
                        get_member_by_path(event, event_type['x_path']),
                        get_member_by_path(event, event_type['y_path']),
                        padded_x,
                        padded_y,
                    )

                    if int(event_x / 16) == x and int(event_y / 16) == y:
                        return event_type, event_idx
        return None

    @if_header_loaded
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        target = self._get_event_by_event_position(event)
        if target is not None:
            event_type, event_idx = target

            # Pick this event as new selection
            tab: EventTab = self.event_widget.tabs[event_type['datatype']]  # type: ignore
            assert isinstance(tab, EventTab), f'Expected EventTab, got {type(tab)}'
            self.event_widget.tab_widget.setCurrentWidget(tab)
            tab.idx_combobox.setCurrentIndex(event_idx)

            # Drag it until the mouse button is released
            self.last_drag = x, y
            # Do not start a macro unless the event is dragged at least one block
            self.drag_started = False
            self.dragged_event = event_type, event_idx
            return

    @if_header_loaded
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drag_started:
                # End a macro only if the event has at least moved one block
                self.event_widget.undo_stack.endMacro()
            self.drag_started = False
            self.dragged_event = None
            self.last_drag = None

    @if_header_loaded
    def mouseDoubleClickEvent(self, event: 'QGraphicsSceneMouseEvent') -> None:
        """Event handler for double click events."""
        if event.button() == Qt.MouseButton.LeftButton:
            target = self._get_event_by_event_position(event)
            if target is not None:
                event_type, event_idx = target
                tab: EventTab = self.event_widget.tabs[event_type['datatype']]  # type: ignore
                assert isinstance(tab, EventTab), f'Expected EventTab, got {type(tab)}'
                tab.goto_header(event_idx)
