""" Module to model mapfooters. """

from . import event
import json
import agb.types
from . import backend

# Define the map block bitfield type
map_block_type = agb.types.BitfieldType('u16', [
    ('block_idx', None, 10),
    ('level', None, 6)
])

# Define a map footer type
footer_type = agb.types.PriorizedMembersStructure([
    ('width', agb.types.u32),
    ('height', agb.types.u32),
    # The borders will be indexed [y][x]
    ('border', agb.types.PointerType(
        agb.types.ArrayType(
            agb.types.ArrayType(
                map_block_type
                ,(lambda parents: int(parents[-2]['border_width']))
            ),
            (lambda parents: int(parents[-1]['border_height']))    
        ), (lambda parents: ('border', 2, False))
    )),
    ('blocks', agb.types.PointerType(
        agb.types.ArrayType(
            agb.types.ArrayType(
                map_block_type
                ,(lambda parents: int(parents[-2]['width']))
            ),
            (lambda parents: int(parents[-1]['height']))    
        ), (lambda parents: ('blocks', 2, False))
    )),
    ('tileset_primary', backend.tileset_pointer_type),
    ('tileset_secondary', backend.tileset_pointer_type),
    ('border_width', agb.types.u8),
    ('border_height', agb.types.u8),
    ('field_1A', agb.types.u16)
], 
# Export the width and height of the blocks and border beforehand
priorized_members=['width', 'height', 'border_width', 'border_height'])
