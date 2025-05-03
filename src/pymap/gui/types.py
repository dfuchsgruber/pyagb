"""Specific datatypes that pymap needs."""

from enum import StrEnum, unique
from typing import Any, Literal, Protocol, Sequence, TypeAlias, overload

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


# The opposite direction of each connection type
opposite_connection_direction: dict[ConnectionType, ConnectionType] = {
    ConnectionType.NORTH: ConnectionType.SOUTH,
    ConnectionType.SOUTH: ConnectionType.NORTH,
    ConnectionType.EAST: ConnectionType.WEST,
    ConnectionType.WEST: ConnectionType.EAST,
}


class BlockProtocol(Protocol):
    """A block in a map."""

    @overload
    def __getitem__(self, key: Literal['block_idx']) -> int: ...

    @overload
    def __getitem__(self, key: Literal['level']) -> int: ...


class Block(dict[str, Any], BlockProtocol):
    """A block in a map."""

    ...
