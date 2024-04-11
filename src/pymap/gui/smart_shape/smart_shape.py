"""Base class for smart shapes."""

from typing import Literal, Protocol, overload
from numpy.typing import NDArray
import numpy as np


class SerializedSmartShape(Protocol):
    """A serialized smart shape."""

    @overload
    def __getitem__(self, key: Literal['type']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['blocks']) -> list[list[int]]:
        ...


class SmartShapeTemplate:
    """Base class for all potential smart shape templates."""


class SmartShape:
    """Base class for smart shape realizations."""
