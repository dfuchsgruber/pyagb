"""Polygon auto-shape template."""

import importlib.resources as resources
from enum import IntEnum
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve  # type: ignore

from pymap.gui.smart_shape.template.generate_blocks import Adjacent, adjacency_kernel
from pymap.gui.types import Tilemap

from ..polygon.polygon import SmartShapeTemplatePolygon


class SmartShapeTemplatePolygonDoubleHeight(SmartShapeTemplatePolygon):
    """Polygon smart shape template with two layers."""

    # TODO: could be generalized to N layers if needed

    class DoubleHeightBlocks(IntEnum):
        """Enumerate the blocks in this template."""

        # Row-wise in a 5x4 grid

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

        UPPER_CORNER_SOUTH_WEST = 10
        UPPER_EDGE_SOUTH = 11
        UPPER_CORNER_SOUTH_EAST = 12
        EDGE_WEST_OVER_LOWER = 13
        EDGE_EAST_OVER_LOWER = 14

        LOWER_CORNER_SOUTH_WEST = 15
        LOWER_EDGE_SOUTH = 16
        LOWER_CORNER_SOUTH_EAST = 17
        UPPER_CORNER_SOUTH_WEST_OVER_LOWER = 18
        UPPER_CORNER_SOUTH_EAST_OVER_LOWER = 19

    @property
    def _template_image_path(self) -> str:
        return str(
            resources.files(
                'pymap.gui.smart_shape.template.polygon_double_height'
            ).joinpath('template.png')
        )

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
        # adjacency_cross = adjacency & Adjacent.ALL_CROSS
        blocks = super().adjacency_flags_to_block(adjacency, mask)

        # Account for the double height
        is_south_edge = (
            (blocks == self.Blocks.EDGE_SOUTH)
            | (blocks == self.Blocks.CORNER_SOUTH_WEST)
            | (blocks == self.Blocks.CORNER_SOUTH_EAST)
        ) & mask

        adjacency_south_edge_cross = (
            cast(
                Tilemap,
                convolve(
                    is_south_edge.astype(int),
                    adjacency_kernel,
                    mode='constant',
                    cval=0,
                ),
            )
            & Adjacent.ALL_CROSS
        )

        block_mapping_dict: dict[int, int] = {
            self.Blocks.CORNER_NORTH_WEST: (self.DoubleHeightBlocks.CORNER_NORTH_WEST),
            self.Blocks.EDGE_NORTH: (self.DoubleHeightBlocks.EDGE_NORTH),
            self.Blocks.CORNER_NORTH_EAST: (self.DoubleHeightBlocks.CORNER_NORTH_EAST),
            self.Blocks.INNER_NORTH_WEST: (self.DoubleHeightBlocks.INNER_NORTH_WEST),
            self.Blocks.INNER_NORTH_EAST: (self.DoubleHeightBlocks.INNER_NORTH_EAST),
            self.Blocks.EDGE_WEST: (self.DoubleHeightBlocks.EDGE_WEST),
            self.Blocks.INNER: (self.DoubleHeightBlocks.INNER),
            self.Blocks.EDGE_EAST: (self.DoubleHeightBlocks.EDGE_EAST),
            self.Blocks.INNER_SOUTH_WEST: (self.DoubleHeightBlocks.INNER_SOUTH_WEST),
            self.Blocks.INNER_SOUTH_EAST: (self.DoubleHeightBlocks.INNER_SOUTH_EAST),
            self.Blocks.CORNER_SOUTH_WEST: (
                self.DoubleHeightBlocks.LOWER_CORNER_SOUTH_WEST
            ),
            self.Blocks.EDGE_SOUTH: (self.DoubleHeightBlocks.LOWER_EDGE_SOUTH),
            self.Blocks.CORNER_SOUTH_EAST: (
                self.DoubleHeightBlocks.LOWER_CORNER_SOUTH_EAST
            ),
        }
        block_mapping_ndarray = np.array(
            [
                block_mapping_dict.get(i, -1)
                for i in range(max(SmartShapeTemplatePolygon.Blocks) + 1)
            ]
        )
        blocks_remapped = block_mapping_ndarray[blocks]
        assert np.all(blocks_remapped != -1), 'Some blocks could not be remapped'

        # # Account for the double height and re-map certain blocks
        # 1. East / west borders may intersect with the second level
        blocks_remapped[
            (blocks_remapped == self.DoubleHeightBlocks.EDGE_WEST)
            & (adjacency_south_edge_cross == (Adjacent.SOUTH))
        ] = self.DoubleHeightBlocks.UPPER_CORNER_SOUTH_WEST
        blocks_remapped[
            (blocks_remapped == self.DoubleHeightBlocks.EDGE_EAST)
            & (adjacency_south_edge_cross == (Adjacent.SOUTH))
        ] = self.DoubleHeightBlocks.UPPER_CORNER_SOUTH_EAST
        blocks_remapped[
            (blocks_remapped == self.DoubleHeightBlocks.INNER)
            & (adjacency_south_edge_cross == (Adjacent.SOUTH))
        ] = self.DoubleHeightBlocks.UPPER_EDGE_SOUTH
        # 2. East / west that does not intersect; just stays east / west border

        # 3. Inner north
        blocks_remapped[
            (blocks_remapped == self.Blocks.INNER_NORTH_WEST)
            & (adjacency_south_edge_cross == (Adjacent.EAST))
        ] = self.DoubleHeightBlocks.EDGE_EAST_OVER_LOWER
        blocks_remapped[
            (blocks_remapped == self.Blocks.INNER_NORTH_EAST)
            & (adjacency_south_edge_cross == (Adjacent.WEST))
        ] = self.DoubleHeightBlocks.EDGE_WEST_OVER_LOWER
        # 4. Inner north with edge to south
        blocks_remapped[
            (blocks_remapped == self.Blocks.INNER_NORTH_WEST)
            & (adjacency_south_edge_cross == (Adjacent.EAST | Adjacent.SOUTH))
        ] = self.DoubleHeightBlocks.UPPER_CORNER_SOUTH_EAST_OVER_LOWER
        blocks_remapped[
            (blocks_remapped == self.Blocks.INNER_NORTH_EAST)
            & (adjacency_south_edge_cross == (Adjacent.WEST | Adjacent.SOUTH))
        ] = self.DoubleHeightBlocks.UPPER_CORNER_SOUTH_WEST_OVER_LOWER

        return blocks_remapped
