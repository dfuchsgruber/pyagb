"""Handles the blocks layer in the map view."""

from pymap.gui.render import tile

from .layer import MapViewLayerTilemap


class MapViewLayerBlocks(MapViewLayerTilemap):
    """A layer in the map view that displays blocks."""

    def load_map(self) -> None:
        """Loads the blocks into the scene."""
        assert self.view.main_gui.block_images is not None
        self.set_rgba_image(
            tile(self.view.main_gui.block_images, self.view.visible_blocks[..., 0])
        )

    def update_block_image_at_padded_position(self, x: int, y: int):
        """Updates the block image at the given padded position.

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
            # Draw the blocks
            block_idx: int = self.view.visible_blocks[y, x, 0]
            assert self.view.main_gui.block_images is not None, 'Blocks are not loaded'
            self.update_rectangle_with_image(
                self.view.main_gui.block_images[block_idx], 16 * x, 16 * y
            )
            self.view.scene().update(
                16 * x,
                16 * y,
                16,
                16,
            )
