"""Handles the events in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType

from .layer import MapViewLayer

if TYPE_CHECKING:
    from .map_view import MapView

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
)

from pymap.gui import properties
from pymap.gui.map.tabs.events.event_image import EventImage
from pymap.gui.render import ndarray_to_QImage


class MapViewLayerEvents(MapViewLayer):
    """A layer in the map view that displays events."""

    def __init__(self, view: MapView):
        """Initialize the layer for events."""
        super().__init__(view)
        self.event_items: dict[str, list[QGraphicsItem]] = {}

    def _build_event_items(self) -> list[QGraphicsItem]:
        """Returns all event items."""
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        event_items: list[QGraphicsItem] = []
        self.event_items: dict[str, list[QGraphicsItem]] = {}
        for event_type in self.view.main_gui.project.config['pymap']['header'][
            'events'
        ].values():
            self.event_items[event_type['datatype']] = []
            events = self.view.main_gui.get_events(event_type)
            for event in events:
                item = self._event_to_qgraphics_item(event, event_type)
                item.setAcceptHoverEvents(False)
                event_items.append(item)
                self.event_items[event_type['datatype']].append(item)
        return event_items

    def load_map(self) -> None:
        """Loads the events into the scene."""
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        events_group = QGraphicsItemGroup()
        for event in self._build_event_items():
            events_group.addToGroup(event)
        events_group.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.item = events_group

    def _event_to_qgraphics_item(
        self, event: ModelValue, event_type: PymapEventConfigType
    ) -> QGraphicsItem:
        """Converts an event to a QGraphicsItem.

        Args:
            event (ModelValue): The event.
            event_type (PymapEventConfigType): The event type.

        Returns:
            QGraphicsItem | None: The QGraphicsItem or None if no image is available.
        """
        assert self.view.main_gui.project is not None
        event_image = self.view.main_gui.project.backend.event_to_image(
            event,
            event_type,
        )
        padded_x, padded_y = self.view.main_gui.get_border_padding()
        x, y = self.view.pad_coordinates(
            properties.get_member_by_path(event, event_type['x_path']),
            properties.get_member_by_path(event, event_type['y_path']),
            padded_x,
            padded_y,
        )
        if event_image is not None and self.show_event_items:
            event_image = EventImage(*event_image)
            pixmap = QPixmap.fromImage(ndarray_to_QImage(event_image.image))
            item = QGraphicsPixmapItem(pixmap)
            item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
            item.setPos(
                16 * (x) + event_image.x_offset, 16 * (y) + event_image.y_offset
            )
        else:
            item = self._get_event_image_rectangle(event_type)
            item.setPos(16 * x, 16 * y)
        return item

    def update_event_images(self):
        """Updates all event images by recomputing them all."""
        if self.item is None:
            return
        group: QGraphicsItemGroup = cast(QGraphicsItemGroup, self.item)
        for event in self.event_items.values():
            for item in event:
                group.removeFromGroup(item)
                self.view.scene().removeItem(item)
        for event in self._build_event_items():
            group.addToGroup(event)

    def update_event_image(self, event_type: PymapEventConfigType, event_idx: int):
        """Updates a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
        """
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        if self.item is None:
            return
        group: QGraphicsItemGroup = cast(QGraphicsItemGroup, self.item)
        item_old = self.event_items[event_type['datatype']][event_idx]
        if item_old in group.childItems():
            group.removeFromGroup(item_old)
            self.view.scene().removeItem(item_old)
            del item_old
        event = self.view.main_gui.get_event(event_type, event_idx)
        item_new = self._event_to_qgraphics_item(event, event_type)
        item_new.setAcceptHoverEvents(False)
        group.addToGroup(item_new)
        self.event_items[event_type['datatype']][event_idx] = item_new

    @property
    def show_event_items(self) -> bool:
        """Returns whether the event images are shown."""
        return cast(
            bool,
            self.view.main_gui.settings.value('event_widget/show_pictures', True, bool),
        )

    def remove_event_image(self, event_type: PymapEventConfigType, event_idx: int):
        """Removes a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
        """
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        if self.item is None:
            return
        group: QGraphicsItemGroup = cast(QGraphicsItemGroup, self.item)
        item = self.event_items[event_type['datatype']][event_idx]
        if item in group.childItems():
            group.removeFromGroup(item)
            self.view.scene().removeItem(item)
            del item
        self.event_items[event_type['datatype']].pop(event_idx)

    def insert_event_image(
        self, event_type: PymapEventConfigType, event_idx: int, event: ModelValue
    ):
        """Inserts a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
            event (ModelValue): The event.
        """
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        if self.item is None:
            return
        group: QGraphicsItemGroup = cast(QGraphicsItemGroup, self.item)
        item = self._event_to_qgraphics_item(event, event_type)
        item.setAcceptHoverEvents(False)
        group.addToGroup(item)
        self.event_items[event_type['datatype']].insert(event_idx, item)

    @staticmethod
    def _get_event_image_rectangle(
        event_type: PymapEventConfigType,
    ) -> QGraphicsItemGroup:
        """Creates a rectangle for the event image.

        Args:
            event_type (PymapEventConfigType): The event type.

        Returns:
            QGraphicsItemGroup: The group.
        """
        group = QGraphicsItemGroup()
        color = QColor.fromRgbF(*(event_type['box_color']))
        rect = QGraphicsRectItem(0, 0, 16, 16)
        rect.setBrush(QBrush(color))
        rect.setPen(QPen(0))
        text = QGraphicsTextItem(event_type['display_letter'][0])
        text.setPos(
            6 - text.sceneBoundingRect().width() / 2,
            6 - text.sceneBoundingRect().height() / 2,
        )

        font = QFont('Ubuntu')
        font.setBold(True)
        font.setPixelSize(16)
        text.setFont(font)
        text.setDefaultTextColor(QColor.fromRgbF(*(event_type['text_color'])))
        group.addToGroup(rect)
        group.addToGroup(text)
        return group
