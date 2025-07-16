"""Handles the the rectangle for the selected event."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymap.configuration import PymapEventConfigType

from .layer import MapViewLayer

if TYPE_CHECKING:
    pass


class MapViewLayerSelectedEvent(MapViewLayer):
    """A layer in the map view that displays the selected event."""

    def load_map(self) -> None:
        """Loads the selected event into the scene."""
        ...

    def update_selected_event_image(
        self, event_type: PymapEventConfigType, event_idx: int | None
    ):
        """Updates the selected event image (the red rectangle).

        Args:
            event_type (PymapEventConfigType): The event type.
            event_idx (int): The event index. If None or -1, the event is hidden.
        """
        ...
