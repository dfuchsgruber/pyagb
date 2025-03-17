"""Polygon auto-shape template."""

import importlib.resources as resources
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve  # type: ignore
from skimage.measure import label as label_image  # type: ignore

from pymap.gui.smart_shape.template.generate_blocks import Adjacent, adjacency_kernel

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
        super().__init__(
            str(
                resources.files('pymap.gui.smart_shape.template.polygon').joinpath(
                    'template.png'
                )
            )
        )

    def generate_blocks(
        self,
        smart_shape: SmartShape,
        map_blocks: NDArray[np.int_],
    ) -> tuple[NDArray[np.int_], tuple[NDArray[np.int_], ...]]:
        """Generate the blocks for the smart shape.

        Args:
            smart_shape (SmartShape): The smart shape
            map_blocks (NDArray[np.int_]): The map blocks

        Returns:
            tuple[NDArray[np.int_], NDArray[np.bool_]]: The blocks and the mask
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
            adjacency: NDArray[np.int_] = convolve(
                mask_label.astype(int),
                adjacency_kernel[::-1, ::-1],
                mode='constant',
                cval=0,  # type: ignore
            )
            converted = np.vectorize(self.adjacency_flags_to_block)(adjacency)
            blocks[converted >= 0] = converted[converted >= 0]
            mask |= (converted >= 0) & mask_label

        idx = np.where(mask)
        blocks = smart_shape.blocks.flatten()[blocks[idx]]

        return blocks, idx

    @classmethod
    def adjacency_flags_to_block(cls, adjacency: Adjacent) -> int:  # noqa: C901
        """Convert the adjacency flags to a block.

        Args:
            adjacency (Adjacent): The adjacency

        Returns:
            int: The block. -1 if not found.
        """
        adjacency_cross = adjacency & Adjacent.ALL_CROSS
        if adjacency_cross == Adjacent.ALL_CROSS:
            # All sides are connected
            if not (adjacency & Adjacent.NORTH_EAST):
                return cls.Blocks.INNER_SOUTH_WEST
            if not (adjacency & Adjacent.SOUTH_EAST):
                return cls.Blocks.INNER_NORTH_WEST
            if not (adjacency & Adjacent.SOUTH_WEST):
                return cls.Blocks.INNER_NORTH_EAST
            if not (adjacency & Adjacent.NORTH_WEST):
                return cls.Blocks.INNER_SOUTH_EAST
            return cls.Blocks.INNER
        # Three adjacent in the cross neighborhood
        elif adjacency_cross == Adjacent.SOUTH | Adjacent.EAST | Adjacent.WEST:
            return cls.Blocks.EDGE_NORTH
        elif adjacency_cross == Adjacent.NORTH | Adjacent.EAST | Adjacent.WEST:
            return cls.Blocks.EDGE_SOUTH
        elif adjacency_cross == Adjacent.NORTH | Adjacent.SOUTH | Adjacent.WEST:
            return cls.Blocks.EDGE_EAST
        elif adjacency_cross == Adjacent.NORTH | Adjacent.SOUTH | Adjacent.EAST:
            return cls.Blocks.EDGE_WEST
        # Two adjacent in the cross neighborhood
        elif adjacency_cross == Adjacent.WEST | Adjacent.SOUTH:
            return cls.Blocks.CORNER_NORTH_EAST
        elif adjacency_cross == Adjacent.WEST | Adjacent.NORTH:
            return cls.Blocks.CORNER_SOUTH_EAST
        elif adjacency_cross == Adjacent.EAST | Adjacent.SOUTH:
            return cls.Blocks.CORNER_NORTH_WEST
        elif adjacency_cross == Adjacent.EAST | Adjacent.NORTH:
            return cls.Blocks.CORNER_SOUTH_WEST
        else:
            return -1
