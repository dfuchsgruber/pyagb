"""Handles the block levels layer in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
)

from pymap.gui.render import tile

from .layer import MapViewLayerTilemap

if TYPE_CHECKING:
    from .map_view import MapView


class MapViewLayerLevels(MapViewLayerTilemap):
    """A layer in the map view that displays levels."""

    def __init__(self, view: MapView):
        """Initialize the layer for levels."""
        super().__init__(view)
        self.level_image_opacity_effect = QGraphicsOpacityEffect()

    def load_map(self) -> None:
        """Loads the levels into the scene."""
        assert self.view.visible_blocks is not None, 'Blocks are not loaded'
        self.set_rgba_image(
            tile(
                self.view.main_gui.map_widget.levels_tab.levels_blocks_rgba,
                self.view.visible_blocks[..., 1],
            )
        )
        assert isinstance(self.item, QGraphicsPixmapItem), (
            'Levels layer is not a pixmap item'
        )
        self.item.setGraphicsEffect(self.level_image_opacity_effect)
        self.update_level_image_opacity()

    def update_level_image_opacity(self):
        """Updates the opacity of the level images."""
        self.level_image_opacity_effect.setOpacity(
            self.view.main_gui.map_widget.levels_tab.level_opacity_slider.sliderPosition()
            / 20
        )

    def update_level_image_at_padded_position(self, x: int, y: int):
        """Updates the level image at the given padded position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        ...
