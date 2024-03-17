"""Utility for computing maps with borders and padding."""


from typing import Sequence

import numpy as np
from agb.model.type import ModelValue
from numpy.typing import NDArray

from pymap.gui.types import Block, UnpackedConnection, ConnectionType
from pymap.project import Project

from . import properties


def compute_blocks(footer: ModelValue, project: Project) -> NDArray[np.int_]:
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


def filter_visible_connections(
    connections: list[UnpackedConnection | None], keep_invisble: bool = False
) -> list[UnpackedConnection | None]:
    """Filters connections for visible and uniqueness in direction.

    Only considers connection types 'north', 'south', 'east' and 'west'.

    Parameters:
    -----------
    connections : list
        A list of connections.
    keep_invisible : bool
        If True, each invisible connection (may it be to its type or invalid data)
        is included as None into the array.

    Returns:
    --------
    filtered : list
        Filtered list of connections, only visible ones are included.
    """
    processed_directions: set[ConnectionType] = set()
    filtered: list[UnpackedConnection | None] = []
    for connection in connections:
        if connection is not None:
            if connection.type not in processed_directions and connection.type in (
                ConnectionType.NORTH,
                ConnectionType.SOUTH,
                ConnectionType.EAST,
                ConnectionType.WEST,
            ):
                processed_directions.add(connection.type)
                filtered.append(connection)
                continue
        if keep_invisble:
            filtered.append(None)
    return filtered


def insert_connection(
    blocks: NDArray[np.int_],
    connection: UnpackedConnection | None,
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
    if (
        connection.type in (ConnectionType.NORTH, ConnectionType.SOUTH)
        and padded_width + connection.offset < 0
    ):
        connection_blocks = connection.blocks[
            :, -(padded_width + connection.offset) :, :
        ]
        offset = -padded_width
    elif (
        connection.type in (ConnectionType.EAST, ConnectionType.WEST)
        and padded_height + connection.offset < 0
    ):
        connection_blocks = connection.blocks[
            -(padded_height + connection.offset) :, :, :
        ]
        offset = -padded_height
    else:
        return

    # Get the segment
    match connection.type:
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


def unpack_connection(
    connection: ModelValue, project: Project, connection_blocks: NDArray[np.int_] | None
) -> UnpackedConnection | None:
    """Loads a connections data if possible.

    Returns:
    --------
    connection_type : int
        The connection type
    offset : int
        The connection offset
    bank : str
        The target bank
    map_idx : str
        The target map_idx
    blocks : ndarray
        The blocks of the connection
    """
    connection_type = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_type_path'],
    )
    assert isinstance(
        connection_type, str
    ), f'Expected str, got {type(connection_type)}'
    offset = properties.get_member_by_path(
        connection,  # type: ignore
        project.config['pymap']['header']['connections']['connection_offset_path'],
    )
    try:
        offset = int(str(offset), 0)
    except ValueError:
        return None  # Non-integer offsets can not be rendered
    bank = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_bank_path'],
    )
    assert isinstance(bank, str), f'Expected str, got {type(bank)}'
    map_idx = properties.get_member_by_path(
        connection,
        project.config['pymap']['header']['connections']['connection_map_idx_path'],
    )
    assert isinstance(map_idx, str), f'Expected str, got {type(map_idx)}'
    try:
        connection_type = int(str(connection_type), 0)
    except ValueError:
        pass
    assert isinstance(
        connection_type, int
    ), f'Expected int, got {type(connection_type)}'
    connection_type = project.config['pymap']['header']['connections'][
        'connection_types'
    ].get(connection_type, None)
    if connection_type in (
        ConnectionType.NORTH,
        ConnectionType.SOUTH,
        ConnectionType.EAST,
        ConnectionType.WEST,
    ):
        # Load the map blocks
        if connection_blocks is None:
            header, _, _ = project.load_header(bank, map_idx)
            if header is None:
                return None
            footer_label = properties.get_member_by_path(
                header, project.config['pymap']['header']['footer_path']
            )
            assert isinstance(
                footer_label, str
            ), f'Expected str, got {type(footer_label)}'
            footer, _ = project.load_footer(footer_label)
            connection_blocks_model = properties.get_member_by_path(
                footer, project.config['pymap']['footer']['map_blocks_path']
            )
            connection_blocks = blocks_to_ndarray(connection_blocks_model)  # type: ignore
        return UnpackedConnection(
            type=connection_type,
            offset=offset,
            bank=bank,
            map_idx=map_idx,
            blocks=connection_blocks,
        )
    return None


def unpack_connections(
    connections: ModelValue,
    project: Project,
    default_blocks: NDArray[np.int_] | None = None,
) -> list[UnpackedConnection | None]:
    """Unpacks a list of connections."""
    assert isinstance(connections, list), f'Expected list, got {type(connections)}'
    return [
        unpack_connection(connection, project, connection_blocks=default_blocks)
        for connection in connections
    ]


def blocks_to_ndarray(blocks: Sequence[Sequence[Block]]) -> NDArray[np.int_]:
    """Converts a blocks list into a numpy ndarray."""
    return np.array(
        [[[data['block_idx'], data['level']] for data in line] for line in blocks],
        dtype=int,
    )


def ndarray_to_blocks(x: NDArray[np.int_]) -> list[list[Block]]:
    """Converts a numpy ndarray back into serializable blocks."""
    return [
        [Block(block_idx=int(block_idx), level=int(level)) for block_idx, level in line]
        for line in x
    ]
