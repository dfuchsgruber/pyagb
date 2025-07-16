"""Handles the grid for the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .layer import MapViewLayer

if TYPE_CHECKING:
    pass


class MapViewLayerGrid(MapViewLayer):
    """A layer in the map view that displays the grid."""

    def load_map(self) -> None:
        """Loads the grid into the scene."""
        ...

    def update_grid(self):
        """Updates the grid."""
        ...
