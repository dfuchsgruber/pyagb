"""Utility for computing maps with borders and padding."""

from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
)

from agb.model.type import ModelValue
from pymap.gui.properties.utils import delete_member_by_path, set_member_by_path
from pymap.gui.render import BlockImages, ndarray_to_QImage
from pymap.gui.types import (
    Block,
    ConnectionType,
    RGBAImage,
    Tilemap,
)
from pymap.project import Project

from . import properties

CONNECTION_BLOCKS_PATH = ['blocks']


def compute_blocks(footer: ModelValue, project: Project) -> Tilemap:
    """Computes all blocks for a given header and footer including borders.

    Parameters:
    -----------
    footer : dict
        The footer to retrieve all blocks from.
    project : Project
        The underlying map project.

    Returns:
    --------
    blocks : ndarray, shape [height, width, 2]
        The blocks padded with border.
    """
    map_blocks = properties.get_member_by_path(
        footer, project.config['pymap']['footer']['map_blocks_path']
    )
    assert isinstance(map_blocks, np.ndarray)
    map_height, map_width, _ = map_blocks.shape
    padded_width, padded_height = project.config['pymap']['display']['border_padding']
    border_blocks = properties.get_member_by_path(
        footer, project.config['pymap']['footer']['border_path']
    )
    assert isinstance(border_blocks, np.ndarray)
    border_blocks = cast(Tilemap, border_blocks)
    border_height, border_width, _ = border_blocks.shape

    # The border is always aligned with the map, therefore one has to consider a
    # virtual block array that is larger than what acutally is displayed
    virtual_reps_x = int(np.ceil(padded_width / border_width)) + int(
        np.ceil((map_width + padded_width) / border_width)
    )
    virtual_reps_y = int(np.ceil(padded_height / border_height)) + int(
        np.ceil((map_height + padded_height) / border_height)
    )
    blocks = np.tile(border_blocks, (virtual_reps_y, virtual_reps_x, 1))
    x0, y0 = (
        (border_width - (padded_width % border_width)) % border_width,
        (border_height - (padded_height % border_height)) % border_height,
    )
    # Create frame exactly the size of the map and its borders repeated with the border
    # sequence, aligned with the origin of the map
    blocks = blocks[
        y0 : y0 + map_height + 2 * padded_height, x0 : x0 + map_width + 2 * padded_width
    ]
    # Insert the map into this frame
    blocks[
        padded_height : padded_height + map_height,
        padded_width : padded_width + map_width,
    ] = map_blocks
    return blocks


def insert_connection(
    blocks: Tilemap,
    connection: ModelValue,
    footer: ModelValue,
    project: Project,
):
    """Inserts a connection into a border padded block array.

    Parameters:
    -----------
    blocks : ndarray, shape [height, width, 2]
        The blocks to insert the connections into, padded with border.
    connection : str, int, str, str, ndarray or None
        The connection to be inserted or None if not connection should be inserted.
    footer : dict
        The footer of the blocks.
    project : Project
        The underlying pymap project.
    """
    if connection is None:
        return
    connection_type = connection_get_connection_type(connection, project)
    if connection_type is None:
        return
    connection_offset = offset = connection_get_offset(connection, project)
    connection_blocks = connection_get_blocks(connection, project)
    padded_width, padded_height = project.config['pymap']['display']['border_padding']
    map_width = properties.get_member_by_path(
        footer, project.config['pymap']['footer']['map_width_path']
    )
    assert isinstance(map_width, int), f'Expected int, got {type(map_width)}'
    map_height = properties.get_member_by_path(
        footer, project.config['pymap']['footer']['map_height_path']
    )
    assert isinstance(map_height, int), f'Expected int, got {type(map_height)}'
    # Trim the connection if the displacement causes its origin to be out of
    # the visible borders

    if connection_blocks is None or connection_offset is None or offset is None:
        # The connection does not have any blocks
        return
    if (
        connection_type in (ConnectionType.NORTH, ConnectionType.SOUTH)
        and padded_width + connection_offset < 0
    ):
        connection_blocks = connection_blocks[
            :, -(padded_width + connection_offset) :, :
        ]
        offset = -padded_width
    elif (
        connection_type in (ConnectionType.EAST, ConnectionType.WEST)
        and padded_height + connection_offset < 0
    ):
        connection_blocks = connection_blocks[
            -(padded_height + connection_offset) :, :, :
        ]
        offset = -padded_height

    # Get the segment
    match connection_type:
        case ConnectionType.NORTH:
            segment = blocks[:padded_height, padded_width + offset :][::-1]
            visible = connection_blocks[::-1][: segment.shape[0], : segment.shape[1]]
            segment[: visible.shape[0], : visible.shape[1]] = visible
        case ConnectionType.SOUTH:
            segment = blocks[padded_height + map_height :, padded_width + offset :]
            visible = connection_blocks[: segment.shape[0], : segment.shape[1]]
            segment[: visible.shape[0], : visible.shape[1]] = visible
        case ConnectionType.EAST:
            segment = blocks[padded_height + offset :, padded_width + map_width :]
            visible = connection_blocks[: segment.shape[0], : segment.shape[1]]
            segment[: visible.shape[0], : visible.shape[1]] = visible
        case ConnectionType.WEST:
            segment = blocks[padded_height + offset :, :padded_width][:, ::-1]
            visible = connection_blocks[:, ::-1][: segment.shape[0], : segment.shape[1]]
            segment[: visible.shape[0], : visible.shape[1]] = visible
        case _:  # type: ignore
            ...


def connection_get_connection_type(
    connection: ModelValue,
    project: Project,
) -> ConnectionType | None:
    """Returns the connection type of a connection.

    Parameters:
    -----------
    connection : ModelValue
        The connection to get the type from.
    project : Project
        The underlying pymap project.

    Returns:
    --------
    ConnectionType | None
        The connection type or None if not found.
    """
    connection_type = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_type_path'],
    )
    if isinstance(connection_type, int):
        connection_type = str(connection_type)
    try:
        connection_type = str(int(str(connection_type), 0))
    except ValueError:
        pass
    if not isinstance(connection_type, str):
        # The connection type is not a valid string
        return None
    try:
        return ConnectionType(
            project.config['pymap']['header']['connections']['connection_types'].get(
                connection_type, None
            )
        )
    except ValueError:
        # The connection type is not a valid enum value
        return None


def connection_get_offset(
    connection: ModelValue,
    project: Project,
) -> int | None:
    """Returns the offset of a connection.

    Parameters:
    -----------
    connection : ModelValue
        The connection to get the offset from.
    project : Project
        The underlying pymap project.

    Returns:
    --------
    int | None
        The offset or None if not found.
    """
    offset = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_offset_path'],
    )
    try:
        offset = int(str(offset), 0)
    except ValueError:
        return None
    return offset


def connection_get_bank(
    connection: ModelValue,
    project: Project,
) -> str | int | None:
    """Returns the bank of a connection.

    Parameters:
    -----------
    connection : ModelValue
        The connection to get the bank from.
    project : Project
        The underlying pymap project.

    Returns:
    --------
    str | int | None
        The bank or None if not found.
    """
    bank = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_bank_path'],
    )
    assert isinstance(bank, (str, int)), f'Expected str, got {type(bank)}'
    return bank


def connection_get_map_idx(
    connection: ModelValue,
    project: Project,
) -> str | int | None:
    """Returns the map index of a connection.

    Parameters:
    -----------
    connection : ModelValue
        The connection to get the map index from.
    project : Project
        The underlying pymap project.

    Returns:
    --------
    str | int | None
        The map index or None if not found.
    """
    map_idx = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_map_idx_path'],
    )
    assert isinstance(map_idx, (str, int)), f'Expected str, got {type(map_idx)}'
    return map_idx


def connection_get_blocks(
    connection: ModelValue,
    project: Project,
) -> Tilemap | None:
    """Returns the blocks of a connection.

    Parameters:
    -----------
    connection : ModelValue
        The connection to get the blocks from.
    project : Project
        The underlying pymap project.

    Returns:
    --------
    Tilemap
        The blocks.
    """
    try:
        blocks = properties.get_member_by_path(
            connection,
            CONNECTION_BLOCKS_PATH,
        )
    except KeyError:
        # The connection does not have any blocks
        return None
    assert isinstance(blocks, (list, np.ndarray)), (
        f'Expected list or ndarray, got {type(blocks)}'
    )
    return cast(Tilemap, blocks)


def unpack_connection(
    connection: ModelValue,
    project: Project,
    connection_blocks: Tilemap | None = None,
) -> ModelValue:
    """Loads a connections data if possible.

    Returns:
    --------
    connection: ModelValue
        The connection data
    """
    connection_type = connection_get_connection_type(connection, project)
    if connection_type in (
        ConnectionType.NORTH,
        ConnectionType.SOUTH,
        ConnectionType.EAST,
        ConnectionType.WEST,
    ):
        # Load the map blocks
        if connection_blocks is None:
            bank = connection_get_bank(connection, project)
            map_idx = connection_get_map_idx(connection, project)
            if bank is not None and map_idx is not None:
                header, _, _ = project.load_header(bank, map_idx)
                if header is not None:
                    footer_label = properties.get_member_by_path(
                        header, project.config['pymap']['header']['footer_path']
                    )
                    assert isinstance(footer_label, str), (
                        f'Expected str, got {type(footer_label)}'
                    )
                    footer, _, _ = project.load_footer(footer_label)
                    connection_blocks_model = properties.get_member_by_path(
                        footer, project.config['pymap']['footer']['map_blocks_path']
                    )
                    assert isinstance(connection_blocks_model, list), (
                        f'Expected list, got {type(connection_blocks_model)}'
                    )
                    connection_blocks = blocks_to_ndarray(
                        cast(Sequence[Sequence[Block]], connection_blocks_model)  # type: ignore
                    )
        connection_blocks = cast(Tilemap, connection_blocks)
        set_member_by_path(
            connection,
            connection_blocks,
            CONNECTION_BLOCKS_PATH,
        )
    return connection


def unpack_connections(
    connections: ModelValue,
    project: Project,
    default_blocks: Tilemap | None = None,
) -> list[ModelValue]:
    """Unpacks a list of connections."""
    assert isinstance(connections, list), f'Expected list, got {type(connections)}'
    return [
        unpack_connection(connection, project, connection_blocks=default_blocks)
        for connection in connections
    ]


def pack_connection(
    connection: ModelValue,
    project: Project,
) -> ModelValue:
    """Packs a connection into a serializable format."""
    try:
        delete_member_by_path(
            connection,
            CONNECTION_BLOCKS_PATH,
        )
    except KeyError:
        # The connection does not have any blocks, so we can skip this step
        pass
    return connection


def pack_connections(
    connections: list[ModelValue],
    project: Project,
) -> list[ModelValue]:
    """Packs a list of connections into a serializable format."""
    return [pack_connection(connection, project) for connection in connections]


def blocks_to_ndarray(blocks: Sequence[Sequence[Block]]) -> RGBAImage:
    """Converts a blocks list into a numpy ndarray."""
    return np.array(
        [[[data['block_idx'], data['level']] for data in line] for line in blocks],
        dtype=int,
    )


def ndarray_to_blocks(x: RGBAImage) -> list[list[Block]]:
    """Converts a numpy ndarray back into serializable blocks."""
    return [
        [Block(block_idx=int(block_idx), level=int(level)) for block_idx, level in line]
        for line in x
    ]


def block_idxs_to_pixmaps(
    block_idxs: RGBAImage,
    block_images: BlockImages,
) -> NDArray[Any]:
    """Maps an ndarray of block_idxs to an array of pixmaps.

    Args:
        block_idxs (RGBAImage): An array of block idxs to map to pixmaps.
        block_images (BlockImages): The images of blocks.

    Returns:
        NDArray: An array of block pixmaps.
    """
    result = np.empty_like(block_idxs, dtype=object)
    for (y, x), block_idx in np.ndenumerate(block_idxs):
        pixmap = QPixmap.fromImage(ndarray_to_QImage(block_images[block_idx]))
        item = QGraphicsPixmapItem(pixmap)
        item.setPos(x * 16, y * 16)
        result[y, x] = item
    return result
