"""Class for the project-specific backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agb.model.type import ModelValue
    from pymap.configuration import PymapEventConfigType
    from pymap.gui.map.tabs.events import EventImage
    from pymap.project import Project


class ProjectBackend:
    """Base-backend for the project."""

    def __init__(self, project: Project):
        """Initializes the backend with the project."""
        self.project = project

    def event_to_image(
        self,
        event: ModelValue,
        event_type: PymapEventConfigType,
    ) -> EventImage | None:
        """Returns the event to image backend."""
        return None
