import numpy as np

DOWN, RIGHT, LEFT, UP, UNDETERMINED = range(5)

delta_to_dir = {
    (-1, 0) : DOWN,
    (0, -1) : RIGHT,
    (0, 1) : LEFT,
    (1, 0) : UP,
}

direction_pair_to_shape_idx = { # (dir_previous, dir_next) -> shape_idx
    (DOWN, DOWN) : 5,
    (DOWN, RIGHT) : 10,
    (DOWN, LEFT) : 9,
    (RIGHT, RIGHT) : 11,
    (RIGHT, DOWN) : 4,
    (RIGHT, UP) : 12,
    (UP, UP) : 7,
    (UP, RIGHT) : 3,
    (UP, LEFT) : 2,
    (LEFT, LEFT) : 1,
    (LEFT, DOWN) : 0,
    (LEFT, UP) : 8,
}

ERROR_SHAPE_IDX = 14

class SmartPath:
    """ Class to capture a smart path. """

    def __init__(self):
        self.coordinates = []
        self._directions = np.zeros((0,), dtype=int) # Idx 0 is missing as it is always treated as corresponding idx -1
        self.shape_idxs = np.zeros((0,), dtype=int)

    def __contains__(self, x):
        return tuple(x) in self.coordinates
        
    def __len__(self):
        return len(self.coordinates)

    def get_by_path_idx(self, idx):
        idx = idx % len(self.coordinates)
        return self.coordinates[idx], self.shape_idxs[idx]

    def _direction(self, x1, x2):
        """ Determines the direction between two coordinates. """
        delta = np.sign(x1 - x2)
        return delta_to_dir.get(tuple(delta), UNDETERMINED)

    def append(self, coordinate):
        self.coordinates.append(tuple(coordinate))
        self._directions = np.append(self._directions, UNDETERMINED)
        self.shape_idxs = np.append(self.shape_idxs, ERROR_SHAPE_IDX)
        
        if len(self.coordinates) > 1:
            n = len(self.coordinates)
            # Recompute directions for idx and 0
            for idx in (n-1, 0):
                self._directions[idx] = self._direction(np.array(self.coordinates[idx - 1]), np.array(self.coordinates[idx])) # previous -> current
            # Update shapes for idx -1, idx and 0
            for idx in (n-2, n-1, 0):
                dir_prev, dir_next = self._directions[idx], self._directions[(idx + 1) % n]
                self.shape_idxs[idx] = direction_pair_to_shape_idx.get((dir_prev, dir_next), ERROR_SHAPE_IDX)