"""Specific datatypes that pymap needs."""

from enum import StrEnum, unique
from typing import Any, Literal, NamedTuple, Protocol, Sequence, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray


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
    blocks: NDArray[np.int_]


class BlockProtocol(Protocol):
    """A block in a map."""

    @overload
    def __getitem__(self, key: Literal['block_idx']) -> int:
        ...

    @overload
    def __getitem__(self, key: Literal['level']) -> int:
        ...


class Block(dict[str, Any], BlockProtocol):
    """A block in a map."""

    ...


MapLayers: TypeAlias = Sequence[int] | int | NDArray[np.int_]
