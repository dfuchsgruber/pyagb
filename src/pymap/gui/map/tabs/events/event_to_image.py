"""Interface for creating an image from a map event."""

from typing import NamedTuple, Protocol

from agb.model.type import ModelValue
from pymap.configuration import PymapEventConfigType
from pymap.gui.types import RGBAImage
from pymap.project import Project


class EventImage(NamedTuple):
    """An image for a map event."""

    image: RGBAImage
    x_offset: int
    y_offset: int


class EventToImage(Protocol):
    """Protocol for event to image backends."""

    def event_to_image(
        self, event: ModelValue, event_type: PymapEventConfigType, project: Project
    ) -> EventImage | None:
        """Converts an event to an image."""
        ...


class NullEventToImage(EventToImage):
    """Default dummy class to translate events into images."""

    def event_to_image(
        self, event: ModelValue, event_type: PymapEventConfigType, project: Project
    ) -> EventImage | None:
        """Returns None."""
        return None
