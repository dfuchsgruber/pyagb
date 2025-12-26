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
from ..polygon.polygon import SmartShapeTemplatePolygon
from ..polygon.polygon import (
    adjacency_flags_to_block as adjacency_flags_to_block_polygon,
)
from ..template import SmartShapeTemplate


class SmartShapeTemplatePolygonDoubleHeight(SmartShapeTemplate):
    """Polygon smart shape template."""

    class Blocks(IntEnum):
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

    def __init__(self):
        """Initialize the template."""
        super().__init__(self._template_image_path)

    @property
    def _template_image_path(self) -> str:
        return str(
            resources.files(
                'pymap.gui.smart_shape.template.polygon_double_height'
            ).joinpath('template.png')
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
            converted = adjacency_flags_to_block(adjacency, mask_label)
            blocks[(converted >= 0) & mask_label] = converted[
                (converted >= 0) & mask_label
            ]
            # converted = np.zeros_like(adjacency)
            mask |= (converted >= 0) & mask_label

        idx = np.where(mask)
        blocks = smart_shape.blocks.flatten()[blocks[idx]]

        return blocks, idx


def adjacency_flags_to_block(adjacency: Tilemap, mask: NDArray[np.bool_]) -> Tilemap:
    """Convert the adjacency flags to a block.

    Args:
        adjacency (Tilemap): The adjacency
        mask (NDArray[np.bool_]): The mask for which to compute blocks

    Returns:
        Tilemap: The block. -1 if not found.
    """
    # adjacency_cross = adjacency & Adjacent.ALL_CROSS
    blocks = adjacency_flags_to_block_polygon(adjacency)

    # Account for the double height
    is_south_edge = (
        (blocks == SmartShapeTemplatePolygon.Blocks.EDGE_SOUTH)
        | (blocks == SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_WEST)
        | (blocks == SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_EAST)
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
        SmartShapeTemplatePolygon.Blocks.CORNER_NORTH_WEST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.CORNER_NORTH_WEST
        ),
        SmartShapeTemplatePolygon.Blocks.EDGE_NORTH: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_NORTH
        ),
        SmartShapeTemplatePolygon.Blocks.CORNER_NORTH_EAST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.CORNER_NORTH_EAST
        ),
        SmartShapeTemplatePolygon.Blocks.INNER_NORTH_WEST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_WEST
        ),
        SmartShapeTemplatePolygon.Blocks.INNER_NORTH_EAST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_EAST
        ),
        SmartShapeTemplatePolygon.Blocks.EDGE_WEST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_WEST
        ),
        SmartShapeTemplatePolygon.Blocks.INNER: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER
        ),
        SmartShapeTemplatePolygon.Blocks.EDGE_EAST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_EAST
        ),
        SmartShapeTemplatePolygon.Blocks.INNER_SOUTH_WEST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_SOUTH_WEST
        ),
        SmartShapeTemplatePolygon.Blocks.INNER_SOUTH_EAST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_SOUTH_EAST
        ),
        SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_WEST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.LOWER_CORNER_SOUTH_WEST
        ),
        SmartShapeTemplatePolygon.Blocks.EDGE_SOUTH: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.LOWER_EDGE_SOUTH
        ),
        SmartShapeTemplatePolygon.Blocks.CORNER_SOUTH_EAST: (
            SmartShapeTemplatePolygonDoubleHeight.Blocks.LOWER_CORNER_SOUTH_EAST
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
        (blocks_remapped == SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_WEST)
        & (adjacency_south_edge_cross == (Adjacent.SOUTH))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.UPPER_CORNER_SOUTH_WEST
    blocks_remapped[
        (blocks_remapped == SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_EAST)
        & (adjacency_south_edge_cross == (Adjacent.SOUTH))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.UPPER_CORNER_SOUTH_EAST
    blocks_remapped[
        (blocks_remapped == SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER)
        & (adjacency_south_edge_cross == (Adjacent.SOUTH))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.UPPER_EDGE_SOUTH
    # 2. East / west that does not intersect; just stays east / west border

    # 3. Inner north
    blocks_remapped[
        (
            blocks_remapped
            == SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_WEST
        )
        & (adjacency_south_edge_cross == (Adjacent.EAST))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_EAST_OVER_LOWER
    blocks_remapped[
        (
            blocks_remapped
            == SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_EAST
        )
        & (adjacency_south_edge_cross == (Adjacent.WEST))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.EDGE_WEST_OVER_LOWER
    # 4. Inner north with edge to south
    blocks_remapped[
        (
            blocks_remapped
            == SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_WEST
        )
        & (adjacency_south_edge_cross == (Adjacent.EAST | Adjacent.SOUTH))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.UPPER_CORNER_SOUTH_EAST_OVER_LOWER
    blocks_remapped[
        (
            blocks_remapped
            == SmartShapeTemplatePolygonDoubleHeight.Blocks.INNER_NORTH_EAST
        )
        & (adjacency_south_edge_cross == (Adjacent.WEST | Adjacent.SOUTH))
    ] = SmartShapeTemplatePolygonDoubleHeight.Blocks.UPPER_CORNER_SOUTH_WEST_OVER_LOWER

    return blocks_remapped
