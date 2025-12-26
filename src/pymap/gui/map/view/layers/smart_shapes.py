"""Handles the smart shapes layer in the map view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QGraphicsItemGroup, QGraphicsOpacityEffect

from pymap.gui.render import tile
from pymap.gui.rgba_image import QRGBAImage

from .layer import MapViewLayer

if TYPE_CHECKING:
    from pymap.gui.map.view import MapView


class MapViewLayerSmartShapes(MapViewLayer):
    """A layer in the map view that displays smart shapes."""

    def __init__(self, view: MapView):
        """Initialize the layer with an RGBA image."""
        super().__init__(view)
        self._smart_shape_rgba_images: dict[str, QRGBAImage] = {}
        self.block_image_opacity_effect: dict[str, QGraphicsOpacityEffect] = {}

    def update_visible_smart_shape_blocks(self) -> None:
        """Update the visible smart shape blocks for a given smart shape name."""
        name = self.view.map_widget.smart_shapes_tab.current_smart_shape_name
        for layer_name, qrgba_image in self._smart_shape_rgba_images.items():
            qrgba_image.item.setVisible(name == layer_name)

    def load_map(self) -> None:
        """Loads the smart shapes into the scene."""
        self._smart_shape_rgba_images = {}
        self.block_image_opacity_effect = {}
        group = QGraphicsItemGroup()
        assert self.view.main_gui.smart_shapes is not None, (
            'Smart shapes are not loaded'
        )
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        assert self.view.main_gui.project.smart_shape_templates is not None, (
            'Smart shape templates are not loaded'
        )
        for name, smart_shape in self.view.main_gui.smart_shapes.items():
            template = self.view.main_gui.project.smart_shape_templates[
                smart_shape.template
            ]
            image = QRGBAImage(
                tile(
                    template.block_images,
                    smart_shape.buffer[..., 0],
                )
            )
            self.block_image_opacity_effect[name] = QGraphicsOpacityEffect()
            padded_width, padded_height = self.view.main_gui.get_border_padding()
            image.item.setPos(16 * padded_width, 16 * padded_height)
            image.item.setGraphicsEffect(self.block_image_opacity_effect[name])
            self._smart_shape_rgba_images[name] = image
            group.addToGroup(image.item)
        self.update_visible_smart_shape_blocks()
        self.update_block_image_opacity()
        self.item = group

    def update_block_image_opacity(self) -> None:
        """Update the opacity of the block images."""
        for opacity_effect in self.block_image_opacity_effect.values():
            opacity_effect.setOpacity(
                self.view.map_widget.smart_shapes_tab.blocks_opacity_slider.sliderPosition()
                / 20
            )

    def update_smart_shape_block_image_at_padded_position(
        self, smart_shape_name: str, x: int, y: int
    ):
        """Updates the block image at the given padded position.

        Args:
            smart_shape_name (str): The name of the smart shape.
            x (int): The x coordinate.
            y (int): The y coordinate.
        """
        assert self.view.visible_blocks is not None, 'Blocks are not loaded'
        assert self.view.main_gui.project is not None, 'Project is not loaded'
        smart_shape = self.view.main_gui.smart_shapes[smart_shape_name]
        template = self.view.main_gui.project.smart_shape_templates[
            smart_shape.template
        ]

        block_idx: int = self.view.main_gui.smart_shapes[smart_shape_name].buffer[
            y, x, 0
        ]
        assert self.view.main_gui.block_images is not None, 'Blocks are not loaded'
        self._smart_shape_rgba_images[smart_shape_name].set_rectangle(
            template.block_images[block_idx],
            16 * x,
            16 * y,
        )
