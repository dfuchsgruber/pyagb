"""Handles the block levels layer in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QGraphicsPixmapItem,
)

from pymap.gui.render import tile

from .layer import MapViewLayerRGBAImage

if TYPE_CHECKING:
    from ..map_view import MapView


class MapViewLayerLevels(MapViewLayerRGBAImage):
    """A layer in the map view that displays levels."""

    def __init__(self, view: MapView):
        """Initialize the layer for levels."""
        super().__init__(view)
        self.level_image_opacity_effect = None

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
        self.level_image_opacity_effect = QGraphicsOpacityEffect()
        self.item.setGraphicsEffect(self.level_image_opacity_effect)
        self.update_level_image_opacity()

    def update_level_image_opacity(self):
        """Updates the opacity of the level images."""
        if self.level_image_opacity_effect is not None:
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
        assert self.view.visible_blocks is not None, 'Blocks are not loaded'
        padded_width, padded_height = self.view.main_gui.get_border_padding()
        map_width, map_height = self.view.main_gui.get_map_dimensions()
        if x in range(2 * padded_width + map_width) and y in range(
            2 * padded_height + map_height
        ):
            # Draw the level blocks
            level: int = self.view.visible_blocks[y, x, 1]
            self.update_rectangle_with_image(
                self.view.main_gui.map_widget.levels_tab.levels_blocks_rgba[level],
                16 * x,
                16 * y,
            )
            # self.view.scene().update(
            #     16 * x,
            #     16 * y,
            #     16,
            #     16,
            # )
