# Utility module for computing maps with borders and connections

import numpy as np
import json
from . import properties

def compute_blocks(footer, project):
    """ Computes all blocks for a given header and footer including borders. 
    
    Parameters:
    -----------
    header : dict
        The map footer to retrieve blocks from. Border and Map Blocks are
        expected to be ndarrays.
    project : Project
        The underlying map project.

    Returns:
    --------
    blocks : ndarray, shape [height, width, 2]
        The blocks padded with border.
    """
    map_blocks = properties.get_member_by_path(footer, project.config['pymap']['footer']['map_blocks_path'])
    map_height, map_width, _ = map_blocks.shape
    padded_width, padded_height = project.config['pymap']['display']['border_padding']
    border_blocks = properties.get_member_by_path(footer, project.config['pymap']['footer']['border_path'])
    border_height, border_width, _ = border_blocks.shape

    # The border is always aligned with the map, therefore one has to consider a virtual block array that is larger than what acutally is displayed
    virtual_reps_x = (int(np.ceil(padded_width / border_width)) + int(np.ceil((map_width + padded_width) / border_width)))
    virtual_reps_y = (int(np.ceil(padded_height / border_height)) + int(np.ceil((map_height + padded_height) / border_height)))
    blocks = np.tile(border_blocks, (virtual_reps_y, virtual_reps_x, 1))
    x0, y0 = (border_width - (padded_width % border_width)) % border_width, (border_height - (padded_height % border_height)) % border_height
    # Create frame exactly the size of the map and its borders repeated with the border sequence, aligned with the origin of the map
    blocks = blocks[y0:y0 + map_height + 2 * padded_height, x0:x0 + map_width + 2 * padded_width]
    # Insert the map into this frame
    blocks[padded_height:padded_height + map_height, padded_width:padded_width + map_width] = map_blocks
    return blocks

def filter_visible_connections(connections, keep_invisble=False):
    """ Filters connections s.t. there are only visible connections and only one connection per direction. 
    Only considers connection types 'north', 'south', 'east' and 'west'.
    
    Parameters:
    -----------
    connections : list
        A list of connections.
    keep_invisible : bool
        If True, each invisible connection (may it be to its type or invalid data) is included as None into the array.
    
    Returns:
    --------
    filtered : list
        Filtered list of connections, only visible ones are included.
    """
    processed_directions = set()
    filtered = []
    for connection in connections:
        if connection is not None:
            connection_type, offset, bank, map_idx, connection_blocks = connection
            if connection_type not in processed_directions and connection_type in ('north', 'south', 'east', 'west'):
                processed_directions.add(connection_type)
                filtered.append(connection)
                continue
        if keep_invisble:
            filtered.append(None)
    return filtered


def insert_connection(blocks, connection, footer, project):
    """ Inserts a connection into a border padded block array.
    
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
    if connection is None: return
    connection_type, offset, bank, map_idx, connection_blocks = connection
    padded_width, padded_height = project.config['pymap']['display']['border_padding']
    map_width = properties.get_member_by_path(footer, project.config['pymap']['footer']['map_width_path'])
    map_height = properties.get_member_by_path(footer, project.config['pymap']['footer']['map_height_path'])
    # Trim the connection if the displacement causes its origin to be out of the visible borders
    if connection_type in ('north', 'south') and padded_width + offset < 0:
        connection_blocks = connection_blocks[:, -(padded_width + offset) : , :]
        offset = -padded_width
    elif connection_type in ('east', 'west') and padded_height + offset < 0:
        connection_blocks = connection_blocks[-(padded_height + offset) : , :, :]
        offset = -padded_height
    # Get the segment
    if connection_type == 'north':
        segment = blocks[ : padded_height, padded_width + offset : ][::-1]
        visible = connection_blocks[::-1][:segment.shape[0], :segment.shape[1]]
        segment[:visible.shape[0], :visible.shape[1]] = visible
    elif connection_type == 'south':
        segment = blocks[padded_height + map_height : , padded_width + offset : ]
        visible = connection_blocks[:segment.shape[0], :segment.shape[1]]
        segment[:visible.shape[0], :visible.shape[1]] = visible
    elif connection_type == 'east': 
        segment = blocks[padded_height + offset :, padded_width + map_width : ]
        visible = connection_blocks[:segment.shape[0], :segment.shape[1]]
        segment[:visible.shape[0], :visible.shape[1]] = visible
    elif connection_type == 'west' :
        segment = blocks[padded_height + offset :, : padded_width][:,::-1]
        visible = connection_blocks[:,::-1][:segment.shape[0], :segment.shape[1]]
        segment[:visible.shape[0], :visible.shape[1]] = visible

def unpack_connection(connection, project, connection_blocks=None):
    """ Loads a connections data if possible. If any value is not inferable, None is returned.
    
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
    connection_type = properties.get_member_by_path(connection, project.config['pymap']['header']['connections']['connection_type_path'])
    offset = properties.get_member_by_path(connection, project.config['pymap']['header']['connections']['connection_offset_path'])
    try: offset = int(str(offset), 0)
    except ValueError: return None # Non-integer offsets can not be rendered
    bank = properties.get_member_by_path(connection, project.config['pymap']['header']['connections']['connection_bank_path'])
    map_idx = properties.get_member_by_path(connection, project.config['pymap']['header']['connections']['connection_map_idx_path'])
    try: connection_type = int(str(connection_type), 0)
    except ValueError: pass
    connection_type = project.config['pymap']['header']['connections']['connection_types'].get(connection_type, None)
    if connection_type in ('north', 'south', 'east', 'west'):
        # Load the map blocks
        if connection_blocks is None:
            header, _, _ = project.load_header(bank, map_idx)
            if header is None: return None
            footer_label = properties.get_member_by_path(header, project.config['pymap']['header']['footer_path'])
            footer, _ = project.load_footer(footer_label)
            connection_blocks = blocks_to_ndarray(properties.get_member_by_path(footer, project.config['pymap']['footer']['map_blocks_path']))
        return connection_type, offset, str(bank), str(map_idx), connection_blocks
    return None

def unpack_connections(connections, project, default_blocks=None):
    """ Unpacks a list of connections. """
    return [unpack_connection(connection, project, connection_blocks=default_blocks) for connection in connections]

def blocks_to_ndarray(blocks):
    """ Converts a blocks list into a numpy ndarray. """
    return np.array([
        [[data['block_idx'], data['level']] for data in line ] for line in blocks
    ], dtype=int)

def ndarray_to_blocks(x):
    """ Converts a numpy ndarray back into serializable blocks. """
    return [
        [{'block_idx' : int(block_idx), 'level' : int(level)} for block_idx, level in line] for line in x
    ]
    