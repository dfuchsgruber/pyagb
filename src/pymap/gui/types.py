"""Specific datatypes that pymap needs."""

from enum import StrEnum, unique
from typing import Any, Literal, NamedTuple, Protocol, Sequence, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray

# Images are represented as 8-bit RGBA values (4 channels)
RGBAImage: TypeAlias = NDArray[np.uint8]  # Shape ..., 4
# Tilemaps are represented as integer maps
Tilemap: TypeAlias = NDArray[np.int_]
# MapLayers are represented as a sequence of integers or a single integer
MapLayers: TypeAlias = Sequence[int] | int | Tilemap


@unique
class ConnectionType(StrEnum):
    """All connection types."""

    NORTH = 'north'
    SOUTH = 'south'
    EAST = 'east'
    WEST = 'west'


class UnpackedConnection(NamedTuple):
    """A connection between two maps, unpacked from the actual model value."""

    type: str
    offset: int
    bank: str | int
    map_idx: str | int
    blocks: Tilemap


class BlockProtocol(Protocol):
    """A block in a map."""

    @overload
    def __getitem__(self, key: Literal['block_idx']) -> int: ...

    @overload
    def __getitem__(self, key: Literal['level']) -> int: ...


class Block(dict[str, Any], BlockProtocol):
    """A block in a map."""

    ...
