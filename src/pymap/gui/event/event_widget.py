"""Widget for events of the header."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QSplitter,
    QTabWidget,
    QWidget,
)

from pymap.configuration import PymapEventConfigType

from .event_to_image import EventToImage, NullEventToImage
from .map_scene import MapScene
from .tab import EventTab

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class EventWidget(QWidget):
    """Class to model events."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the event widget.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): Parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.undo_stack = QtGui.QUndoStack()

        # Layout is similar to the map widget
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        splitter = QSplitter()
        layout.addWidget(splitter, 1, 1, 1, 1)
        self.map_scene = MapScene(self)
        self.map_scene_view = QtWidgets.QGraphicsView()
        self.map_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.map_scene_view.setScene(self.map_scene)
        splitter.addWidget(self.map_scene_view)
        self.info_label = QtWidgets.QLabel()
        layout.addWidget(self.info_label, 2, 1, 1, 2)
        layout.setRowStretch(1, 1)
        layout.setRowStretch(2, 0)
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.map_scene.update_selection)
        self.tabs = {}
        splitter.addWidget(self.tab_widget)
        splitter.setSizes([4 * 10**6, 10**6])  # Ugly as hell hack to take large values

    @property
    def header_loaded(self) -> bool:
        """Whether the header is loaded.

        Returns:
            bool: Whether the header is loaded.
        """
        return self.main_gui.header is not None and self.main_gui.project is not None

    def load_project(self):
        """Load a new project."""
        assert self.main_gui.project is not None, 'Project is None'
        self.load_header()

        # Create a tab for each event type
        self.tab_widget.clear()
        self.tabs: dict[str, EventTab] = {}

        for event_type in self.main_gui.project.config['pymap']['header']['events']:
            self.tabs[event_type['datatype']] = EventTab(self, event_type)
            self.tab_widget.addTab(
                self.tabs[event_type['datatype']], event_type['name']
            )

        # Load backend for event_to_image
        project = self.main_gui.project
        backend = project.config['pymap']['display']['event_to_image_backend']

        if backend is not None:
            with self.main_gui.project.project_dir():
                with open(Path(backend)) as f:
                    namespace: dict[str, object] = {}
                    exec(f.read(), namespace)
                    get_event_to_image = namespace['get_event_to_image']
                    assert isinstance(get_event_to_image, Callable)
                    self.event_to_image: EventToImage = cast(
                        EventToImage, get_event_to_image()
                    )
        else:
            self.event_to_image = NullEventToImage()

    def load_header(self):
        """Opens a new header."""
        self.map_scene.clear()
        self.load_map()
        self.load_events()

    def load_map(self):
        """Reloads the map image by using tiles of the map widget."""
        if self.main_gui.project is None or self.main_gui.header is None:
            return

        # Load pixel maps directly from the map widget
        for (y, x), item in np.ndenumerate(self.main_gui.map_widget.map_images):
            pixmap = item.pixmap()
            item = QGraphicsPixmapItem(pixmap)  # Create a new item bound to this canvas
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)

        # Load rectangles directly from the map widget
        border_color = QColor.fromRgbF(
            *(self.main_gui.project.config['pymap']['display']['border_color'])
        )
        assert self.main_gui.map_widget.north_border is not None
        self.map_scene.addRect(
            self.main_gui.map_widget.north_border.rect(),
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        assert self.main_gui.map_widget.south_border is not None
        self.map_scene.addRect(
            self.main_gui.map_widget.south_border.rect(),
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        assert self.main_gui.map_widget.east_border is not None
        self.map_scene.addRect(
            self.main_gui.map_widget.east_border.rect(),
            pen=QPen(0),
            brush=QBrush(border_color),
        )

        assert self.main_gui.map_widget.west_border is not None
        self.map_scene.addRect(
            self.main_gui.map_widget.west_border.rect(),
            pen=QPen(0),
            brush=QBrush(border_color),
        )
        self.map_scene.setSceneRect(
            0,
            0,
            16 * self.main_gui.map_widget.map_images.shape[1],
            16 * self.main_gui.map_widget.map_images.shape[0],
        )

    def load_events(self):
        """Loads all event images."""
        for datatype in self.tabs:
            self.tabs[datatype].load_events()
        self.map_scene.update_selection()

    def update_event(self, event_type: PymapEventConfigType, event_idx: int):
        """Updates a certain event."""
        tab = self.tabs[event_type['datatype']]

        # Recalculate the group of this event
        self.map_scene.removeItem(
            self.map_scene.event_groups[event_type['datatype']][event_idx]
        )
        event = self.main_gui.get_event(event_type, event_idx)

        group = tab.event_to_group(event)
        self.map_scene.addItem(group)
        self.map_scene.event_groups[event_type['datatype']][event_idx] = group

        if tab.idx_combobox.currentIndex() == event_idx:
            # Update the properties tree
            tab.event_properties.update()
            if self.tab_widget.currentWidget() is tab:
                # Update the selection
                self.map_scene.update_selection()
