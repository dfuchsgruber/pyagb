# Module to draw maps

import agb.image
from PIL import Image
from warnings import warn
from properties import get_member_by_path
import numpy as np

def get_border(footer, blocks, project):
    """ Computes the image of the border.
    
    Parameters:
    -----------
    footer : dict
        The map footer.
    blocks : list
        A list of 16x16 blocks.
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    border_img : PIL.Image
        An image of the border.
    """
    width = int(get_member_by_path(footer, project.config['pymap']['footer']['border_width_path']))
    height = int(get_member_by_path(footer, project.config['pymap']['footer']['border_height_path']))
    border_blocks = get_member_by_path(footer, project.config['pymap']['footer']['border_path'])
    border_img = Image.new('RGBA', (width * 16, height * 16))
    for y, block_line in enumerate(border_blocks):
        for x, block_data in enumerate(block_line):
            block_idx = block_data[0]
            border_img.paste(blocks[block_idx], (16 * x, 16 * y), blocks[block_idx])
    return border_img

# height x width x level static array that enumerates all blocks
block_map = np.array([[idx, 0] for idx in range(0x400)]).reshape((-1, 8, 2))

def draw_blocks(blocks):
    """ Draws a picture of all available blocks. 
    
    Parameters:
    -----------
    blocks : list
        A lsit of 16x16 blocks

    Returns:
    --------
    block_img : PIL.Image
        An image of all blocks.
    """
    return draw_block_map(blocks, block_map)

def draw_block_map(blocks, block_map):
    """ Computes the image of an entire block map.
    
    Parameters:
    -----------
    blocks : list
        A list of 16x16 blocks.
    block_map : np.array, shape [H, W, >=1]
        The blocks in a 2d array.

    Returns:
    --------
    map_img : PIL.Image
        An image of the entire map, size (16*H, 16*W).
    """
    height, width, _ = block_map.shape
    map_img = Image.new('RGBA', (width * 16, height * 16))
    for (y, x), block_idx in np.ndenumerate(block_map[:, :, 0]):
        map_img.paste(blocks[block_idx], (16 * x, 16 * y), blocks[block_idx])
    return map_img

# Depcreated
def get_map(footer, blocks):
    """ Computes the image of an entire map.
    
    Parameters:
    -----------
    footer : dict
        The map footer.
    blocks : list
        A list of 16x16 blocks.

    Returns:
    --------
    map_img : PIL.Image
        An image of the entire map.
    """
    width = int(get_member_by_path(footer, project.config['pymap']['footer']['map_width_path']))
    height = int(get_member_by_path(footer, project.config['pymap']['footer']['map_height_path']))
    map_blocks = get_member_by_path(footer, project.config['pymap']['footer']['map_blocks_path'])
    map_img = Image.new('RGBA', (width * 16, height * 16))
    for y, block_line in enumerate(map_blocks):
        for x, block_data in enumerate(block_line):
            block_idx = block_data[0]
            map_img.paste(blocks[block_idx], (16 * x, 16 * y), blocks[block_idx])
    return map_img

def get_blocks(tileset_primary, tileset_secondary, tiles, project):
    """ Computes the set of blocks for a combination of tilesets. 
    
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
    blocks_primary = get_member_by_path(tileset_primary, project.config['pymap']['tileset_primary']['blocks_path'])
    blocks_secondary = get_member_by_path(tileset_secondary, project.config['pymap']['tileset_secondary']['blocks_path'])
    return [get_block(block, tiles, project) for block in blocks_primary + blocks_secondary]

def get_block(block, tiles, project):
    """ Computes a block. 
    
    Parameters:
    -----------
    block : dict
        The block to build.
    tiles : list
        All tiles in all palettes.
    project : pymap.project.Project
        The pymap project.
    
    Returns:
    --------
    image : PIL.Image, size 16x16
        The Pilow image of the block.
    """
    block_img = Image.new('RGBA', (16, 16), 'black')
    for idx, data in enumerate(block):
        x, y = idx & 1, (idx >> 1) & 1
        pal_idx = data['palette_idx']
        if pal_idx >= 13:
            warn(f'Pal index >= 13: {pal_idx}')
            pal_idx = 0
        tile_idx = data['tile_idx']
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



def get_tile(image_primary, image_secondary, packed_palettes_primary, packed_palettes_secondary, tile_idx, pal_idx, project):
    """ Retrieves a tile from the tilesets in a certain palette.
    
    Parameters:
    -----------
    image_primary : agb.image
        The gfx of the primary tileset.
    image_seconary : agb.image
        The gfx of the secondary tileset.
    packed_palette_primary : list
        A list of packed colors for the primary tileset.
    packed_palette_secondary : list
        A list of packed colors for the secondary tileset.
    tile_idx : int
        The tile index.
    pal_idx : int
        The palette index.
    project : pymap.project.Project
        The pymap project.
    
    Returns:
    --------
    image : PIL.Image, size 8x8
        The Pilow image of the tile.
    """
    assert(tile_idx >= 0 and tile_idx < 0x400)
    assert(pal_idx >= 0 and pal_idx <= 12)
    if pal_idx < 7: colors = packed_palettes_primary[pal_idx]
    else: colors = packed_palettes_secondary[pal_idx - 7]

def get_tiles(tileset_primary, tileset_secondary, project):
    """ Computes the gfx of both tilesets in all possible palettes and splits it into 8x8 tiles. 
    
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
    tiles : list
        A list with #palettes entries. Each list contains a list of at most 0x400 tiles
        of size 8x8.
    """

    # Load images
    gfx_primary = get_member_by_path(tileset_primary, project.config['pymap']['tileset_primary']['gfx_path'])
    gfx_secondary = get_member_by_path(tileset_secondary, project.config['pymap']['tileset_secondary']['gfx_path'])
    gfx_primary = project.load_gfx(True, gfx_primary)
    gfx_secondary = project.load_gfx(False, gfx_secondary)

    # Pack colors for both tilesets
    palettes_primary = get_member_by_path(tileset_primary, project.config['pymap']['tileset_primary']['palettes_path'])
    palettes_secondary = get_member_by_path(tileset_secondary, project.config['pymap']['tileset_secondary']['palettes_path'])

    palettes = pack_colors(palettes_primary, project) + pack_colors(palettes_secondary, project)
    return [
        split_into_tiles(gfx_primary.to_pil_image(palette, transparent=0)) + split_into_tiles(gfx_secondary.to_pil_image(palette, transparent=0)) for palette in palettes
    ]

def split_into_tiles(image):
    """ Splits an image into 8x8 tiles. """
    width, height = image.size
    tiles = []
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            tiles.append(image.crop((x, y, x + 8, y + 8)))
    return tiles


def pack_colors(palettes, project):
    """ Packs all members of a palette structure.
    
    Parameters:
    -----------
    palettes : list
        A palette array.
    project : pymap.project.Project
        The pymap project.
    
    Returns:
    --------
    packed : list
        A list of length #palettes that holds packed rgb values (lists of size 48) for the palette.
    """
    packed = []
    for palette in palettes:
        # Retrieve rgb values separately
        packed_palette = []
        assert(len(palette) == 16) # Only allow 16 colors
        for color in palette:
            for channel in ('red', 'blue', 'green'):
                packed_palette.append(color[channel] << 3)
        packed.append(packed_palette)
    return packed

def select_blocks(blocks, x0, x1, y0, y1):
    """ Helper method to select a subset of an array by a box that may be negative. 
    
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