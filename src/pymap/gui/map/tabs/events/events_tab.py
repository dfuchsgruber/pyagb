"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6 import QtWidgets
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QSplitter,
    QTabWidget,
    QWidget,
)

from pymap.configuration import PymapEventConfigType
from pymap.gui.map_scene import MapScene

from ..tab import MapWidgetTab
from .tab import EventTab

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class EventsTab(MapWidgetTab):
    """Tabs with block like functionality.

    They have a selection of blocks that can be drawn to the map. They also
    support flood filling and replacement. The selection can be taken from the map as
    well or set natively via the `set_selection` method to `selection`.
    """

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

    def map_scene_mouse_pressed(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates
                (with border padding).
            y (int): The y coordinate of the mouse in map coordinates
                (with border padding).
        """

    def map_scene_mouse_moved(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """

    def map_scene_mouse_released(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
