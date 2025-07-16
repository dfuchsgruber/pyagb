"""Handles the smart shapes layer in the map view."""

from .layer import MapViewLayerTilemap


class MapViewLayerSmartShapes(MapViewLayerTilemap):
    """A layer in the map view that displays smart shapes."""

    def load_map(self) -> None:
        """Loads the smart shapes into the scene."""
        ...
