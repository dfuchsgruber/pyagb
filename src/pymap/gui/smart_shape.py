"""Generates smart shapes from a path."""

from enum import IntEnum, auto
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Coordinate: TypeAlias = tuple[int, int]


class SmartShapeDirection(IntEnum):
    """Enum to capture the direction of a smart shape."""

    DOWN = auto()
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    UNDETERMINED = auto()


delta_to_dir: dict[Coordinate, SmartShapeDirection] = {
    (-1, 0): SmartShapeDirection.DOWN,
    (0, -1): SmartShapeDirection.RIGHT,
    (0, 1): SmartShapeDirection.LEFT,
    (1, 0): SmartShapeDirection.UP,
}

direction_pair_to_shape_idx: dict[
    tuple[SmartShapeDirection, SmartShapeDirection], int
] = {  # (dir_previous, dir_next) -> shape_idx
    (SmartShapeDirection.DOWN, SmartShapeDirection.DOWN): 5,
    (SmartShapeDirection.DOWN, SmartShapeDirection.RIGHT): 10,
    (SmartShapeDirection.DOWN, SmartShapeDirection.LEFT): 9,
    (SmartShapeDirection.RIGHT, SmartShapeDirection.RIGHT): 11,
    (SmartShapeDirection.RIGHT, SmartShapeDirection.DOWN): 4,
    (SmartShapeDirection.RIGHT, SmartShapeDirection.UP): 12,
    (SmartShapeDirection.UP, SmartShapeDirection.UP): 7,
    (SmartShapeDirection.UP, SmartShapeDirection.RIGHT): 3,
    (SmartShapeDirection.UP, SmartShapeDirection.LEFT): 2,
    (SmartShapeDirection.LEFT, SmartShapeDirection.LEFT): 1,
    (SmartShapeDirection.LEFT, SmartShapeDirection.DOWN): 0,
    (SmartShapeDirection.LEFT, SmartShapeDirection.UP): 8,
}

ERROR_SHAPE_IDX = 14


class SmartPath:
    """Class to capture a smart path."""

    def __init__(self):
        """Initializes the smart path."""
        self.coordinates: list[tuple[int, int]] = []
        self._directions = np.zeros(
            (0,), dtype=int
        )  # Idx 0 is missing as it is always treated as corresponding idx -1
        self.shape_idxs = np.zeros((0,), dtype=int)

    def __contains__(self, x: Coordinate) -> bool:
        """Checks if a coordinate is in the smart path.

        Args:
            x (Coordinate): The coordinate to check.

        Returns:
            bool: True if the coordinate is in the smart path.
        """
        return tuple(x) in self.coordinates

    def __len__(self) -> int:
        """Returns the length of the smart path.

        Returns:
            int: The length of the smart path.
        """
        return len(self.coordinates)

    def get_by_path_idx(self, idx: int) -> tuple[Coordinate, int]:
        """Gets a coordinate and its shape index by path index.

        Args:
            idx (int): The path index.

        Returns:
            tuple[Coordinate, int]: The coordinate and its shape index.
        """
        idx = idx % len(self.coordinates)
        return self.coordinates[idx], self.shape_idxs[idx]

    def _direction(
        self, x1: npt.NDArray[np.int_], x2: npt.NDArray[np.int_]
    ) -> SmartShapeDirection:
        """Determines the direction between two coordinates."""
        delta = np.sign(x1 - x2)
        return delta_to_dir.get(tuple(delta), SmartShapeDirection.UNDETERMINED)  # type: ignore

    def append(self, coordinate: Coordinate | npt.NDArray[np.int_]):
        """Appends a coordinate to the smart path.

        Args:
            coordinate (Coordinate | npt.NDArray[np.int_]): The coordinate to append.
        """
        self.coordinates.append(tuple(coordinate))  # type: ignore
        self._directions = np.append(self._directions, SmartShapeDirection.UNDETERMINED)
        self.shape_idxs = np.append(self.shape_idxs, ERROR_SHAPE_IDX)

        if len(self.coordinates) > 1:
            n = len(self.coordinates)
            # Recompute directions for idx and 0
            for idx in (n - 1, 0):
                self._directions[idx] = self._direction(
                    np.array(self.coordinates[idx - 1]), np.array(self.coordinates[idx])
                )  # previous -> current
            # Update shapes for idx -1, idx and 0
            for idx in (n - 2, n - 1, 0):
                dir_prev, dir_next = (
                    self._directions[idx],
                    self._directions[(idx + 1) % n],
                )
                self.shape_idxs[idx] = direction_pair_to_shape_idx.get(
                    (dir_prev, dir_next), ERROR_SHAPE_IDX
                )
