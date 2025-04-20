"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui.icon import Icon, icon_paths
from pymap.gui.map_scene import MapScene
from pymap.gui.properties.utils import get_member_by_path

from ..tab import MapWidgetTab
from .properties import ConnectionProperties

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class ConnectionsTab(MapWidgetTab):
    """Tab for the events on the map."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)

        # Layout is similar to the map widget
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.mirror_offset = QtWidgets.QCheckBox('Mirror Displacement to Adjacent Map')
        self.mirror_offset.setChecked(
            self.map_widget.main_gui.settings.value(
                'connections/mirror_offset', True, bool
            )  # type: ignore
        )
        self.mirror_offset.stateChanged.connect(self.mirror_offset_changed)
        self.idx_combobox = QtWidgets.QComboBox()
        self.add_button = QtWidgets.QPushButton()
        self.add_button.setIcon(QIcon(icon_paths[Icon.PLUS]))
        self.add_button.clicked.connect(self.append_connection)
        self.remove_button = QtWidgets.QPushButton()
        self.remove_button.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        self.remove_button.clicked.connect(
            lambda: self.remove_connection(self.idx_combobox.currentIndex())
        )
        self.connection_properties = ConnectionProperties(self)
        self.open_connection = QtWidgets.QPushButton('Open adjacent map')
        self.open_connection.clicked.connect(self.open_adjacent_map)
        self.idx_combobox.currentIndexChanged.connect(self.select_connection)

        layout.addWidget(self.mirror_offset, 1, 1, 1, 3)
        layout.addWidget(self.idx_combobox, 2, 1)
        layout.addWidget(self.open_connection, 4, 1, 1, 3)
        layout.addWidget(self.connection_properties, 3, 1, 1, 3)
        layout.addWidget(self.remove_button, 2, 3)
        layout.addWidget(self.add_button, 2, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)

        # Recording mouse events
        self.last_dragged_position: tuple[int, int] | None = None
        self.dragged_connection = None
        self.is_dragging = False

    @property
    def connection_loaded(self) -> bool:
        """Returns whether a connection is loaded."""
        return (
            self.map_widget.main_gui.project is not None
            and self.map_widget.main_gui.header is not None
        )

    def mirror_offset_changed(self):
        """Event handler for when the mirror offset checkbox is toggled."""
        self.map_widget.main_gui.settings.setValue(
            'connections/mirror_offset', self.mirror_offset.isChecked()
        )

    def load_project(self):
        """Loads the project."""

    def load_header(self):
        """Loads the tab."""

    @property
    def visible_layers(self) -> MapScene.VisibleLayer:
        """Get the visible layers."""
        return (
            MapScene.VisibleLayer.BLOCKS
            | MapScene.VisibleLayer.BORDER_EFFECT
            | MapScene.VisibleLayer.CONNECTIONS
            | MapScene.VisibleLayer.SELECTED_EVENT
        )

    def _get_connection_by_padded_position(
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

    def _select_connection_at_padded_position(
        self,
        x: int,
        y: int,
    ):
        """Selects an event at the given position."""
        target = self._get_connection_by_padded_position(x, y)
        if target is not None:
            ...
        else:
            # If no event is selected, set the dragged event to None
            self.last_dragged_position = None
            self.dragged_connection = None
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
        self._select_connection_at_padded_position(x, y)

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_connection_at_padded_position(x, y)

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """
        self._select_connection_at_padded_position(x, y)

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
            assert self.dragged_connection is not None, 'Dragged event is None'
            ...
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
        self.dragged_connection = None

    def map_scene_mouse_double_clicked(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for double clicking the mouse."""
        if not self.map_widget.header_loaded:
            return
        if event.buttons() != Qt.MouseButton.LeftButton:
            return
        target = self._get_connection_by_padded_position(x, y)
        if target is None:
            return
        # TODO: go to connecting map

    def append_connection(self) -> None:
        """Append a new connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        # TODO

    def remove_connection(self, index: int) -> None:
        """Remove a connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        # TODO

    def open_adjacent_map(self) -> None:
        """Open the adjacent map."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        # TODO

    def select_connection(self) -> None:
        """Select a connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        # TODO

    def update_connection(self, connection_idx: int, mirror_offset: bool):
        """Update the connection."""
        if not self.map_widget.header_loaded:
            return
        if not self.map_widget.main_gui.project:
            return
        # TODO
