"""Module for rendering blocks and tiles to PIL Images."""

from typing import TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt
from PySide6.QtGui import QImage, qRgb

from agb.image import Image, Palette
from agb.model.type import ModelValue
from pymap.project import Project

from .properties import get_member_by_path

TileImages: TypeAlias = npt.NDArray[
    np.uint8
]  # Shape: num_palettes x num_tiles x 8 x 8 x 4
BlockImages: TypeAlias = npt.NDArray[np.uint8]  # Shape: num_blocks x 16 x 16 x 4
PackedPalette: TypeAlias = list[int]


# height x width x level static array that enumerates all blocks
blocks_pool = np.array([[idx, 0] for idx in range(0x400)], dtype=np.uint8).reshape(
    (128, 8, 2)
)


def draw_blocks_pool(block_images: BlockImages) -> npt.NDArray[np.uint8]:
    """Draws a picture of all available blocks.

    Parameters:
    -----------
    block_images : np.ndarray, shape [num_blocks, 16, 16, 4]
        A list of 16x16 blocks.

    Returns:
    --------
    block_img : np.ndarray, shape [128 * 16, 8 * 16, 4]
    """
    return draw_blocks(block_images, blocks_pool[..., 0])


def draw_blocks(
    block_images: BlockImages, block_map: npt.NDArray[np.uint8] = blocks_pool[..., 0]
) -> npt.NDArray[np.uint8]:
    """Computes the image of an entire block map.

    Parameters:
    -----------
    blocks : BlockImages, shape [num_blocks, 16, 16, 4]
        A list of 16x16 blocks.
    block_map : np.array, shape [H, W]
        The blocks in a 2d array.

    Returns:
    --------
    map_img : np.ndarray, shape [H * 16, W * 16, 4]
        An image of the tilemap.
    """
    return tile(block_images, block_map)


def get_blocks(
    tileset_primary: ModelValue,
    tileset_secondary: ModelValue,
    tiles: TileImages,
    project: Project,
) -> npt.NDArray[np.uint8]:
    """Computes the set of blocks for a combination of tilesets.

    Parameters:
    -----------
    tileset_primary : dict
        The primary tileset.
    tileset_secondary : dict
        The seocndary tileset.
    tiles : list
        All tiles in all palettes.
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    blocks : list
        A list of 16x16 block images.
    """
    blocks_primary = get_member_by_path(
        tileset_primary, project.config['pymap']['tileset_primary']['blocks_path']
    )
    blocks_secondary = get_member_by_path(
        tileset_secondary, project.config['pymap']['tileset_secondary']['blocks_path']
    )
    assert isinstance(blocks_primary, list)
    assert isinstance(blocks_secondary, list)
    blocks = np.stack(
        [get_block(block, tiles) for block in blocks_primary + blocks_secondary], axis=0
    )
    return blocks


def get_block(block: ModelValue, tiles: TileImages) -> npt.NDArray[np.uint8]:
    """Computes a block.

    Parameters:
    -----------
    block : ModelValue
        The block to build.
    tiles : np.ndarray, shape [num_palettes, num_tiles, 8, 8, 4]
        All tiles in all palettes.

    Returns:
    --------
    image : np.ndarray, shape [16, 16, 4]
        The image of the block.
    """
    block_img = np.zeros((16, 16, 4), dtype=np.uint8)
    assert isinstance(block, list)
    for idx, data in enumerate(block):
        x, y = idx & 1, (idx >> 1) & 1
        assert isinstance(data, dict)
        pal_idx = data['palette_idx']
        assert isinstance(pal_idx, int)
        if pal_idx >= 13:
            warn(f'Pal index >= 13: {pal_idx}')
            pal_idx = 0
        tile_idx = data['tile_idx']
        assert isinstance(tile_idx, int)
        if tile_idx >= 1024:
            warn(f'Tile index >= 1024: {tile_idx}')
            tile_idx = 0
        tile = tiles[pal_idx, tile_idx]
        if data['horizontal_flip']:
            tile = tile[:, ::-1]
        if data['vertical_flip']:
            tile = tile[::-1]
        # We do not properly realize the alpha channel
        # and instead say 0 is transparent
        mask = tile[..., 3] != 0
        block_img[
            8 * y : 8 * (y + 1),
            8 * x : 8 * (x + 1),
        ][mask] = tile[mask]
    return block_img


def get_tiles(
    tileset_primary: ModelValue, tileset_secondary: ModelValue, project: Project
) -> TileImages:
    """A list if all tiles in all possible palettes.

    Parameters:
    -----------
    tileset_primary : dict
        The primary tileset.
    tileset_secondary : dict
        The seocndary tileset.
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    tiles : TileImages
        A list with #palettes entries. Each list contains a list of at most 0x400 tiles
        of size 8x8.
    """
    # Load images
    gfx_primary = get_member_by_path(
        tileset_primary, project.config['pymap']['tileset_primary']['gfx_path']
    )
    gfx_secondary = get_member_by_path(
        tileset_secondary, project.config['pymap']['tileset_secondary']['gfx_path']
    )
    assert isinstance(gfx_primary, str)
    assert isinstance(gfx_secondary, str)
    gfx_primary = project.load_gfx(True, gfx_primary)
    gfx_secondary = project.load_gfx(False, gfx_secondary)

    # Pack colors for both tilesets
    palettes_primary = get_member_by_path(
        tileset_primary, project.config['pymap']['tileset_primary']['palettes_path']
    )
    palettes_secondary = get_member_by_path(
        tileset_secondary, project.config['pymap']['tileset_secondary']['palettes_path']
    )
    palettes = np.concatenate(
        [pack_colors(palettes_primary), pack_colors(palettes_secondary)], axis=0
    )
    return np.stack(
        [
            np.concatenate(
                [
                    split_image_into_tiles(gfx.to_rgba(palette, transparent=0)).reshape(
                        -1, 8, 8, 4
                    )
                    for gfx in (gfx_primary, gfx_secondary)
                ],
                axis=0,
            )
            for palette in palettes
        ]
    )


def split_image_into_tiles(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Splits an image into 8x8 tiles that are cropped from the image.

    Args:
        image (np.ndarray, shape [h, w, 4]): The image to split.

    Returns:
        np.ndarray, shape [h // 8, w // 8, 8, 8, 4]: The tiles.
    """
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 4
    h, w, _ = image.shape
    assert h % 8 == 0
    assert w % 8 == 0
    tiles = np.zeros((h // 8, w // 8, 8, 8, 4), dtype=np.uint8)
    # TODO: efficiency, can we improve the tiling runtime?
    for i in range(h // 8):
        for j in range(w // 8):
            tiles[i, j] = image[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
    return tiles


def tile(
    tiles: npt.NDArray[np.uint8], tilemap: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Tiles a list of tiles into a single image.

    Args:
        tiles (np.ndarray, shape [num_tiles, t1, ... tN, ...]): The tiles to tile.
        tilemap (np.ndarray, shape [m1, ..., mN]): The tilemap to use.

    Returns:
        tiled: np.ndarray, shape [m1 * t1, ..., mN * tN, ...]: The tiled image.

    """
    output = np.zeros(
        tuple(i * j for i, j in zip(tilemap.shape, tiles.shape[1:]))
        + tuple(tiles.shape[1 + len(tilemap.shape) :]),
        dtype=tiles.dtype,
    )
    for idxs in np.ndindex(tilemap.shape):
        output[
            tuple(slice(i * j, (i + 1) * j) for i, j in zip(idxs, tiles.shape[1:]))
        ] = tiles[tilemap[idxs]]
    return output


def pack_colors(palettes: ModelValue) -> npt.NDArray[np.uint8]:
    """Packs all members of a palette structure.

    Parameters:
    -----------
    palettes : list
        A palette array.

    Returns:
    --------
    packed : np.ndarray, shape [num_palettes, 16, 3]
    """
    assert isinstance(palettes, list)
    return np.array(
        [
            [
                [
                    8 * color[channel]  # type: ignore
                    for channel in ('red', 'blue', 'green')
                ]
                for color in palette
                if isinstance(color, dict)
            ]
            for palette in palettes
            if isinstance(palette, list)
        ],
        dtype=np.uint8,
    )


def ndarray_to_QImage(
    array: npt.NDArray[np.uint8],
) -> QImage:
    """Generates a QImage from a numpy array.

    Args:
        array (npt.NDArray[np.uint8]): The array to convert, shape [h, w, 4].

    Returns:
        QImage: The QImage.
    """
    H, W, C = array.shape
    if C != 4:
        raise ValueError('Array must have 4 channels!')

    # Convert to uint8 if necessary
    if array.dtype == np.uint8:
        array_u8 = array
    elif array.dtype == np.float32 or array.dtype == np.float64:
        warn('Converting float array to uint8')
        array_u8 = (array * 255).clip(0, 255).astype(np.uint8)
    elif np.issubdtype(array.dtype, np.integer):
        warn('Converting int array to uint8')
        array_u8 = np.clip(array, 0, 255).astype(np.uint8)
    else:
        raise ValueError('Unsupported data type for image array')

    # Create QImage
    return QImage(array_u8.data, W, H, 4 * W, QImage.Format.Format_RGBA8888)


def image_to_QImage(image: Image, palette: Palette) -> QImage:
    """Creates a QImage from a GBA image."""
    qimage = QImage(
        image.width,
        image.height,
        QImage.Format.Format_Indexed8,
    )
    qimage.setColorCount(len(palette))
    for i in range(len(palette)):
        qimage.setColor(i, qRgb(*palette[i].rgbs))  # type: ignore

    assert image.width % 8 == 0, 'Image width must be a multiple of 8!'
    assert image.height % 8 == 0, 'Image height must be a multiple of 8!'
    buffer = qimage.bits()
    buffer[:] = bytes(image.data.flatten())  # type: ignore[assignment]

    return qimage


def select_blocks[E: np.generic](
    blocks: npt.NDArray[E], x0: int, x1: int, y0: int, y1: int
) -> npt.NDArray[E]:
    """Helper method to select a subset of an array by a box that may be negative.

    Parameters:
    -----------
    blocks : ndarray, shape [h, w]
        The blocks to select from.
    x0 : int
        X-coorindate of the first corner.
    y0 : int
        Y-coordinate of the first corner.
    x1 : int
        X-coordinate of the second corner.
    y1 : int
        Y-coordinate of the second_corner

    Returns:
    --------
    selected : ndarray, shape [|x0 - x1|, |y0 - y1|]
        The selection described by the box.
    """
    if x1 <= x0:
        x0, x1 = x1 - 1, x0 + 1
    if y1 <= y0:
        y0, y1 = y1 - 1, y0 + 1
    return blocks[y0:y1, x0:x1]


def get_box(x0: int, x1: int, y0: int, y1: int) -> tuple[int, int, int, int]:
    """Helper method to fix a negative box.

    Parameters:
    ------------
    x0 : int
        X-coorindate of the first corner.
    y0 : int
        Y-coordinate of the first corner.
    x1 : int
        X-coordinate of the second corner.
    y1 : int
        Y-coordinate of the second_corner

    Returns:
    ------------
    x0 : int
        X-coorindate of the first corner.
    y0 : int
        Y-coordinate of the first corner.
    x1 : int
        X-coordinate of the second corner.
    y1 : int
        Y-coordinate of the second_corner
    """
    if x1 <= x0:
        x0, x1 = x1 - 1, x0 + 1
    if y1 <= y0:
        y0, y1 = y1 - 1, y0 + 1
    return x0, x1, y0, y1
