"""Module for a transparent background tile."""

import importlib.resources as resources

import numpy as np

from agb.image import from_file
from pymap.gui.render import tile
from pymap.gui.types import RGBAImage, Tilemap


def get_transparent_background(width: int, height: int) -> RGBAImage:
    """Creates a transparent background tile.

    Args:
        width (int): The width of the tile.
        height (int): The height of the tile.

    Returns:
        RGBAImage: The transparent background tile.
    """
    # Create a transparent image
    image, palette = from_file(
        str(resources.files('pymap.gui.transparent').joinpath('transparent_tile.png'))
    )
    img = image.to_rgba(palette.to_pil_palette(), transparent=None)
    assert img.shape == (8, 8, 4), f'Expected image shape {(8, 8, 4)}, got {img.shape}'
    assert width % 8 == 0 and height % 8 == 0, (
        f'Width {width} and height {height} must be multiples of 8'
    )
    tilemap: Tilemap = np.zeros((height // 8, width // 8), dtype=int)
    tiled = tile(img[None, ...], tilemap)
    return tiled
