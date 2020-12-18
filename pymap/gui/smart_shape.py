import numpy as np
from skimage.draw import polygon, polygon_perimeter
from scipy.ndimage import generic_filter

UNMATCHED = 13
INVALID = 14

# If a piece has less than four piece neighbours, it is part of a "convex" edge of the polygon
# We can match the four-neighbour footprint (i.e. which of these neighbours is present) to get
# the idx of the smart shape part
footprints_four_neighbours = np.array([
    [0, 0, 1, 1],
    [0, 1, 1, 1],
    [0, 1, 0, 1],
    # Concave footprints (1)
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    
    [1, 0, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    # Concave footprints (2)
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 0],
])

# Pieces with 8 piece neighbours are are a "concave" edge-connection
# By looking at which of the 9-neighbourhood is not present we identify
# which smart shape idx to put
footprints_five_neighbours = {
    0 : 8,
    2 : 9,
    6 : 4,
    8 : 3,
}

def fooprint_match(x):
    if x[4] == 0:
        return -1 # Not part of the shape
    x_four_neighbours = x[[1, 3, 5, 7]]
    matches = (x_four_neighbours == footprints_four_neighbours).all(1)
    if matches.sum() > 1 and x.sum() == 8:
        return footprints_five_neighbours[x.argmin()]
    if matches.sum() == 1: # Convex piece
        return matches.argmax()
    elif matches.sum() > 1 and x.sum() == 8: # Concave piece
        return footprints_five_neighbours.get(x.argmin(), default=UNMATCHED)
    return INVALID
         
def filled_to_shape(filled, footprints_four_neighbours):
    return 

def path_to_shape(coordinates, auto_shape):
    """ Transforms a path (a boundary of a shape) into a smart shape by placing adequate tiles. 
    
    Parameters:
    -----------
    coordinates : list-like
        A list of (y, x) coordinates where a boundary was set
    auto_shape : ndarray, shape [15]
        Block idxs for each part of the shape.
    """
    # print(list(coordinates))
    coords = np.array(list(coordinates))
    if coords.shape[0] <= 1:
        return np.array([UNMATCHED])
    # coords_unshifted = coords.copy()
    # Create a grid to extract 
    y_min, x_min = coords.min(0)
    y_max, x_max = coords.max(0)
    grid = np.zeros((y_max - y_min + 1, x_max - x_min + 1))
    coords[:, 0] -= y_min
    coords[:, 1] -= x_min
    rr, cc = polygon(coords[:, 0], coords[:, 1], grid.shape)
    grid[rr, cc] = 1
    rr, cc = polygon_perimeter(coords[:, 0], coords[:, 1], grid.shape)
    grid[rr, cc] = 1
    shape_idxs = generic_filter(grid, fooprint_match, (3, 3), mode='constant')
    shape_idxs = shape_idxs[tuple(coords.T)].astype(np.int)
    return auto_shape[shape_idxs]

dirs = np.array([
    [-1, 0], # down,
    [0, -1], # right
    [0, 1], # left
    [1, 0], # up
])

DOWN, RIGHT, LEFT, UP = range(4)

def coordinates_to_directions(coordinates):
    """ Transforms a history of coordinates into directions """
    deltas = np.sign(coordinates[:-1] - coordinates[1:])
    directions = []
    for delta in deltas:
        match = (delta == dirs).all(1)
        if match.any():
            directions.append(match.argmax())
        else:
            directions.append(-1)
    return directions

direction_pair_to_block = {
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

def path_to_shape2(coordinates, auto_shape):
    """ Transforms a sequence of directions into blocks. """
    coordinates = np.array(list(coordinates))
    if coordinates.shape[0] <= 1:
        return np.array([UNMATCHED])
    directions = coordinates_to_directions(coordinates)
    print(directions)
    shape_idxs = np.fromiter((direction_pair_to_block.get((prev, cur), UNMATCHED) for (prev, cur) in zip([directions[0]] + directions[:-1], directions)), int)
    return auto_shape[shape_idxs]

