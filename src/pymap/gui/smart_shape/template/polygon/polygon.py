"""Polygon auto-shape template."""

import importlib.resources as resources
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve  # type: ignore
from skimage.measure import label as label_image  # type: ignore

from pymap.gui.smart_shape.template.generate_blocks import Adjacent, adjacency_kernel
from pymap.gui.types import Tilemap

from ...smart_shape import SmartShape
from ..template import SmartShapeTemplate


class SmartShapeTemplatePolygon(SmartShapeTemplate):
    """Polygon smart shape template."""

    class Blocks(IntEnum):
        """Enumerate the blocks in this template."""

        CORNER_NORTH_WEST = 0
        EDGE_NORTH = 1
        CORNER_NORTH_EAST = 2
        INNER_NORTH_WEST = 3
        INNER_NORTH_EAST = 4
        EDGE_WEST = 5
        INNER = 6
        EDGE_EAST = 7
        INNER_SOUTH_WEST = 8
        INNER_SOUTH_EAST = 9
        CORNER_SOUTH_WEST = 10
        EDGE_SOUTH = 11
        CORNER_SOUTH_EAST = 12

    def __init__(self):
        """Initialize the template."""
        super().__init__(self._template_image_path)

    @property
    def _template_image_path(self) -> str:
        """Get the template path."""
        return str(
            resources.files('pymap.gui.smart_shape.template.polygon').joinpath(
                'template.png'
            )
        )

    def generate_blocks(
        self,
        smart_shape: SmartShape,
        map_blocks: Tilemap,
    ) -> tuple[Tilemap, tuple[NDArray[np.int_], ...]]:
        """Generate the blocks for the smart shape.

        Args:
            smart_shape (SmartShape): The smart shape
            map_blocks (Tilemap): The map blocks

        Returns:
            tuple[Tilemap, NDArray[np.bool_]]: The blocks and the mask
        """
        buffer = smart_shape.buffer[..., 0]
        assert ((0 <= buffer) & (buffer < self.num_blocks)).all(), 'Invalid blocks.'
        blocks = np.full_like(buffer, -1)
        mask = np.zeros_like(buffer, dtype=bool)

        # Find all connected components in the buffer
        labeled = label_image(buffer + 1, connectivity=1)  # type: ignore
        non_background_labels: list[int] = np.unique(labeled[buffer != 0]).tolist()  # type: ignore

        for label in non_background_labels:
            mask_label: NDArray[np.bool_] = labeled == label  # type: ignore
            adjacency: Tilemap = cast(
                Tilemap,
                convolve(
                    mask_label.astype(int),
                    adjacency_kernel,
                    mode='constant',
                    cval=0,  # type: ignore
                ),
            )
            converted = self.adjacency_flags_to_block(adjacency, mask_label)
            blocks[(converted >= 0) & mask_label] = converted[
                (converted >= 0) & mask_label
            ]
            # converted = np.zeros_like(adjacency)
            mask |= (converted >= 0) & mask_label

        idx = np.where(mask)
        blocks = smart_shape.blocks.flatten()[blocks[idx]]

        return blocks, idx

    def adjacency_flags_to_block(
        self, adjacency: Tilemap, mask: NDArray[np.bool_]
    ) -> Tilemap:
        """Convert the adjacency flags to a block.

        Args:
            adjacency (Tilemap): The adjacency
            mask (NDArray[np.bool_]): The mask for which to compute blocks
        Returns:
            Tilemap: The block. -1 if not found.
        """
        adjacency_cross = adjacency & Adjacent.ALL_CROSS
        blocks = np.full_like(adjacency, -1)

        # All sides are connected
        mask = adjacency_cross == Adjacent.ALL_CROSS
        blocks[mask] = SmartShapeTemplatePolygon.Blocks.INNER
        blocks[mask & (~((adjacency & Adjacent.NORTH_EAST) > 0))] = (
            SmartShapeTemplatePolygon.Blocks.INNER_SOUTH_WEST
        )
        blocks[mask & (~((adjacency & Adjacent.SOUTH_EAST) > 0))] = (
            SmartShapeTemplatePolygon.Blocks.INNER_NORTH_WEST
        )
        blocks[mask & (~((adjacency & Adjacent.SOUTH_WEST) > 0))] = (
            SmartShapeTemplatePolygon.Blocks.INNER_NORTH_EAST
        )
        blocks[mask & (~((adjacency & Adjacent.NORTH_WEST) > 0))] = (
            SmartShapeTemplatePolygon.Blocks.INNER_SOUTH_EAST
        )

        # Three adjacent in the cross neighborhood
        blocks[adjacency_cross == (Adjacent.SOUTH | Adjacent.EAST | Adjacent.WEST)] = (
            SmartShapeTemplatePolygon.Blocks.EDGE_NORTH
        )
        blocks[adjacency_cross == (Adjacent.NORTH | Adjacent.EAST | Adjacent.WEST)] = (
            SmartShapeTemplatePolygon.Blocks.EDGE_SOUTH
        )
        blocks[adjacency_cross == (Adjacent.NORTH | Adjacent.SOUTH | Adjacent.WEST)] = (
            SmartShapeTemplatePolygon.Blocks.EDGE_EAST
        )
        blocks[adjacency_cross == (Adjacent.NORTH | Adjacent.SOUTH | Adjacent.EAST)] = (
            SmartShapeTemplatePolygon.Blocks.EDGE_WEST
        )

        # Two adjacent in the cross neighborhood
        blocks[adjacency_cross == (Adjacent.WEST | Adjacent.SOUTH)] = (
            SmartShapeTemplatePolygon.Blocks.CORNER_NORTH_EAST
        )
        blocks[adjacency_cross == (Adjacent.WEST | Adjacent.NORTH)] = (
            SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_EAST
        )
        blocks[adjacency_cross == (Adjacent.EAST | Adjacent.SOUTH)] = (
            SmartShapeTemplatePolygon.Blocks.CORNER_NORTH_WEST
        )
        blocks[adjacency_cross == (Adjacent.EAST | Adjacent.NORTH)] = (
            SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_WEST
        )

        return blocks
