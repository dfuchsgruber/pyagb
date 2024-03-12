"""Specific datatypes that pymap needs."""

from enum import StrEnum, unique
from typing import NamedTuple, Sequence, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


@unique
class ConnectionType(StrEnum):
    """All connection types."""

    NORTH = 'north'
    SOUTH = 'south'
    EAST = 'east'
    WEST = 'west'


class Connection(NamedTuple):
    """A connection between two maps."""

    type: str
    offset: int
    bank: str
    map_idx: str
    blocks: NDArray[np.int_]


class Block(TypedDict):
    """A block in a map."""

    block_idx: int
    level: int


MapLayers: TypeAlias = Sequence[int] | int | NDArray[np.int_]
