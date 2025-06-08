"""Module to compress tilesets by removing reundant tiles."""

import itertools
from typing import NamedTuple, cast

import numpy as np

from agb.image import Image
from agb.model.type import ModelValue
from pymap.gui.properties.utils import get_member_by_path
from pymap.gui.render import split_image_into_tiles, tile
from pymap.gui.types import (
    RGBAImage,
)
from pymap.project import Project


class RemappedTile(NamedTuple):
    """A remapped tile with its index and flip flags."""

    tile_idx: int
    horizontal_flip: bool
    vertical_flip: bool


def get_redundant_tiles(
    tiles: RGBAImage,
    vflips: bool = True,
    hflips: bool = True,
) -> dict[int, RemappedTile]:
    """Finds all tiles that are redundant.

    This includes tiles that are identical when flipped horizontally or vertically.

    Args:
        tiles (RGBAImage): The tiles to check for redundancy.
        vflips (bool, optional): Whether to check for vertical flips.
            Defaults to True.
        hflips (bool, optional): Whether to check for horizontal flips.
            Defaults to True.

    Returns:
        dict[int, RemappedTile]: A mapping of tile indices to their remapped
            counterparts.
    """
    unique_tiles = {}
    remapped: dict[int, RemappedTile] = {}

    for tile_idx, _tile in enumerate(tiles):
        # Each tile is a 8x8 array of color indices
        tile_key = tuple(_tile.flatten())
        for h_flip, v_flip in itertools.product(
            (False, True) if hflips else (False,),
            (False, True) if vflips else (False,),
        ):
            flipped_tile = _tile.copy()
            if h_flip:
                flipped_tile = flipped_tile[:, ::-1]
            if v_flip:
                flipped_tile = flipped_tile[::-1, :]
            tile_key_flipped = tuple(flipped_tile.flatten())
            if tile_key_flipped in unique_tiles:
                remapped[tile_idx] = RemappedTile(
                    tile_idx=unique_tiles[tile_key_flipped],
                    horizontal_flip=h_flip,
                    vertical_flip=v_flip,
                )
                break
        else:
            unique_tiles[tile_key] = tile_idx
    return remapped


def image_remove_redundant_tiles(
    image: Image,
    replace_redundant_tiles: bool = True,
    fill_value: int = 0,
) -> dict[int, RemappedTile]:
    """Removes redundant tiles from an image.

    Args:
        image (Image): The image to process.
        replace_redundant_tiles (bool, optional): Whether to replace redundant tiles in
            the image. Defaults to True.
        fill_value (int, optional): The value to use for filling in removed tiles.
        Defaults to 0.

    Returns:
        dict[int, RemappedTile]: A mapping of tile indices to their remapped
            counterparts.
    """
    tiles_indexed = split_image_into_tiles(image.data.T).reshape(
        -1, 8, 8
    )  # num_tiles, 8, 8 array
    remapping = get_redundant_tiles(
        tiles_indexed,
    )
    # Erase the redundant tiles from the tileset
    if replace_redundant_tiles:
        tiles_indexed[list(remapping.keys())] = fill_value
        tiles = tile(
            tiles_indexed,
            np.arange(len(tiles_indexed)).reshape(-1, image.data.shape[0] // 8),
        )
        image.data = tiles.T
    return remapping


def tileset_remove_redundant_tiles(
    project: Project,
    tileset_label: str,
    is_primary: bool = True,
) -> tuple[dict[int, RemappedTile], dict[str, int]]:
    """Removes redundant tiles from a tileset.

    This function will also update all co-occurring tilesets with the same tileset
    label, ensuring that all tilesets that share the same tiles are updated with the
    new tile mappings.

    Args:
        project (Project): The project containing the tileset.
        tileset_label (str): The label of the tileset to process.
        is_primary (bool, optional): Whether the tileset is primary. Defaults to True.

    Returns:
        tuple[dict[int, RemappedTile], dict[str, int]]: A mapping of tile indices
            to theirremapped counterparts and a mapping of tileset labels to the
            number of updated blocks.
    """
    # First load the tiles of the tileset
    tileset = project.load_tileset(is_primary, tileset_label)
    assert tileset is not None, (
        f'Tileset {tileset_label} not found as '
        f'{"primary" if is_primary else "secondary"}.'
    )
    gfx_label = get_member_by_path(
        tileset,
        project.config['pymap'][
            'tileset_primary' if is_primary else 'tileset_secondary'
        ]['gfx_path'],
    )
    assert isinstance(gfx_label, str), 'Tileset must have a valid gfx label.'
    gfx, palette = project.load_gfx(is_primary, gfx_label, True)
    remapping = image_remove_redundant_tiles(gfx, replace_redundant_tiles=True)
    if not is_primary:
        remapping = {
            k + 0x280: RemappedTile(
                tile_idx=v.tile_idx + 0x280,  # Offset for secondary tiles
                horizontal_flip=v.horizontal_flip,
                vertical_flip=v.vertical_flip,
            )
            for k, v in remapping.items()
        }

    tilesets_primary, tilesets_secondary = project.get_cooccuring_tilesets(
        tileset_label, is_primary
    )
    num_updated_blocks: dict[str, int] = {}
    for other_tileset_label, is_other_primary in [
        (other_tileset_label, True) for other_tileset_label in tilesets_primary
    ] + [(other_tileset_label, False) for other_tileset_label in tilesets_secondary]:
        num_updated_blocks[other_tileset_label] = 0

        other_tileset = project.load_tileset(is_other_primary, other_tileset_label)
        blocks: list[list[dict[str, ModelValue]]] = cast(
            list[list[dict[str, ModelValue]]],
            get_member_by_path(
                other_tileset,
                project.config['pymap'][
                    'tileset_primary' if is_other_primary else 'tileset_secondary'
                ]['blocks_path'],
            ),
        )
        for block in blocks:
            for data in block:
                tile_idx = cast(int, data['tile_idx'])
                assert isinstance(tile_idx, int)
                if tile_idx in remapping:
                    remapped_tile = remapping[tile_idx]
                    data['tile_idx'] = remapped_tile.tile_idx

                    data['horizontal_flip'] = int(
                        bool(remapped_tile.horizontal_flip)
                        != bool(data['horizontal_flip'])
                    )
                    data['vertical_flip'] = int(
                        bool(remapped_tile.vertical_flip) != bool(data['vertical_flip'])
                    )
                    num_updated_blocks[other_tileset_label] += 1
        if num_updated_blocks[other_tileset_label] > 0:
            ...
            project.save_tileset(is_other_primary, other_tileset, other_tileset_label)
    project.save_gfx(is_primary, gfx, palette.to_pil_palette(), gfx_label)
    return remapping, {k: v for k, v in num_updated_blocks.items() if v > 0}
