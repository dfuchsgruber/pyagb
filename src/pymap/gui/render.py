"""Module for rendering blocks and tiles to PIL Images."""

from typing import Sequence, SupportsInt, TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt
from agb.model.type import ModelValue
from PIL import Image

from pymap.project import Project

from .properties import get_member_by_path

TileImages: TypeAlias = list[list[Image.Image]]
BlockImages: TypeAlias = list[Image.Image]
PackedPalette: TypeAlias = list[int]


def get_border(
    footer: ModelValue, block_images: BlockImages, project: Project
) -> Image.Image:
    """Computes the image of the border.

    Parameters:
    -----------
    footer : dict
        The map footer.
    block_images : list
        A list of 16x16 block images.
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    border_img : PIL.Image
        An image of the border.
    """
    width, height = (
        get_member_by_path(footer, project.config['pymap']['footer'][dimension])
        for dimension in ('border_width_path', 'border_width_path')
    )
    assert isinstance(width, SupportsInt)
    assert isinstance(height, SupportsInt)
    width, height = int(width), int(height)
    border_blocks = get_member_by_path(
        footer, project.config['pymap']['footer']['border_path']
    )
    assert isinstance(border_blocks, list)
    border_img = Image.new('RGBA', (width * 16, height * 16))
    for y, block_line in enumerate(border_blocks):
        assert isinstance(block_line, list)
        for x, block_data in enumerate(block_line):
            assert isinstance(block_data, Sequence)
            block_idx = block_data[0]
            assert isinstance(block_idx, int)
            border_img.paste(
                block_images[block_idx], (16 * x, 16 * y), block_images[block_idx]
            )
    return border_img


# height x width x level static array that enumerates all blocks
blocks_pool = np.array([[idx, 0] for idx in range(0x400)]).reshape((-1, 8, 2))


def draw_blocks_pool(block_images: BlockImages) -> Image.Image:
    """Draws a picture of all available blocks.

    Parameters:
    -----------
    blocks : list
        A lsit of 16x16 blocks

    Returns:
    --------
    block_img : PIL.Image
        An image of all blocks.
    """
    return draw_blocks(block_images, blocks_pool)


def draw_blocks(
    block_images: BlockImages, blocks_pool: npt.NDArray[np.int_] = blocks_pool
) -> Image.Image:
    """Computes the image of an entire block map.

    Parameters:
    -----------
    blocks : list
        A list of 16x16 blocks.
    blocks_pool : np.array, shape [H, W, >=1]
        The blocks in a 2d array.

    Returns:
    --------
    map_img : PIL.Image
        An image of the entire map, size (16*H, 16*W).
    """
    height, width, _ = blocks_pool.shape
    map_img = Image.new('RGBA', (width * 16, height * 16))
    for (y, x), block_idx in np.ndenumerate(blocks_pool[:, :, 0]):
        map_img.paste(
            block_images[block_idx], (16 * x, 16 * y), block_images[block_idx]
        )
    return map_img


def get_blocks(
    tileset_primary: ModelValue,
    tileset_secondary: ModelValue,
    tiles: TileImages,
    project: Project,
) -> BlockImages:
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
    return [get_block(block, tiles) for block in blocks_primary + blocks_secondary]


def get_block(block: ModelValue, tiles: TileImages) -> Image.Image:
    """Computes a block.

    Parameters:
    -----------
    block : dict
        The block to build.
    tiles : list
        All tiles in all palettes.

    Returns:
    --------
    image : PIL.Image, size 16x16
        The Pilow image of the block.
    """
    block_img = Image.new('RGBA', (16, 16), 'black')
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
        tile = tiles[pal_idx][tile_idx]
        if data['horizontal_flip']:
            tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
        if data['vertical_flip']:
            tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
        block_img.paste(tile, (8 * x, 8 * y), tile)
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

    palettes = pack_colors(palettes_primary) + pack_colors(palettes_secondary)
    return [
        [
            img
            for row in split_image_into_tiles(
                gfx_primary.to_pil_image(palette, transparent=0)
            )
            for img in row
        ]
        + [
            img
            for row in split_image_into_tiles(
                gfx_secondary.to_pil_image(palette, transparent=0)
            )
            for img in row
        ]
        for palette in palettes
    ]


def split_image_into_tiles(image: Image.Image) -> TileImages:
    """Splits an image into 8x8 tiles.

    Args:
        image (Image.Image): The image to split.

    Returns:
        TileImages: A flat list of 8x8 tiles.
    """
    width, height = image.size
    return [
        [image.crop((x, y, x + 8, y + 8)) for x in range(0, width, 8)]
        for y in range(0, height, 8)
    ]


def pack_colors(palettes: ModelValue) -> list[PackedPalette]:
    """Packs all members of a palette structure.

    Parameters:
    -----------
    palettes : list
        A palette array.

    Returns:
    --------
    packed : list
        A list of length #palettes that holds packed rgb values (lists of size 48)
        for the palette.
    """
    packed: list[PackedPalette] = []
    assert isinstance(palettes, list)
    for palette in palettes:
        assert isinstance(palette, list)
        # Retrieve rgb values separately
        packed_palette: list[int] = []
        assert len(palette) == 16  # Only allow 16 colors
        for color in palette:
            assert isinstance(color, dict)
            for channel in ('red', 'blue', 'green'):
                assert channel in color
                rgb = color[channel]
                assert isinstance(rgb, int)
                packed_palette.append(rgb << 3)
        packed.append(packed_palette)
    return packed


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
