"""Handles the events in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType

from .layer import MapViewLayer

if TYPE_CHECKING:
    pass


class MapViewLayerEvents(MapViewLayer):
    """A layer in the map view that displays events."""

    def load_map(self) -> None:
        """Loads the events into the scene."""
        ...

    def update_event_images(self):
        """Updates all event images by recomputing them all."""
        ...

    def update_event_image(self, event_type: PymapEventConfigType, event_idx: int):
        """Updates a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
        """
        ...

    @property
    def show_event_images(self) -> bool:
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
        ...

    def insert_event_image(
        self, event_type: PymapEventConfigType, event_idx: int, event: ModelValue
    ):
        """Inserts a certain event image.

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index.
            event (ModelValue): The event.
        """
        ...
