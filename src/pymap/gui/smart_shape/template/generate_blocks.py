"""Generate the blocks for a smart shape utility."""

from enum import IntFlag, auto

import numpy as np


class Adjacent(IntFlag):
    """Adjacency directions for a block."""

    NORTH = auto()
    NORTH_EAST = auto()
    EAST = auto()
    SOUTH_EAST = auto()
    SOUTH = auto()
    SOUTH_WEST = auto()
    WEST = auto()
    NORTH_WEST = auto()
    ALL = (
        NORTH | NORTH_EAST | EAST | SOUTH_EAST | SOUTH | SOUTH_WEST | WEST | NORTH_WEST
    )
    ALL_CROSS = NORTH | EAST | SOUTH | WEST


# Kernel for 3x3 adjacency
adjacency_kernel = np.array(
    [
        [Adjacent.NORTH_WEST, Adjacent.NORTH, Adjacent.NORTH_EAST],
        [Adjacent.WEST, 0, Adjacent.EAST],
        [Adjacent.SOUTH_WEST, Adjacent.SOUTH, Adjacent.SOUTH_EAST],
    ],
    dtype=int,
)[::-1, ::-1]  # Flip for convolution
