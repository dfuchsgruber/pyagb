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

border_line_type = agb.types.ArrayType(
    'map_block',
    (lambda project, context, parents: int(parents[-2]['border_width']))
)

border_array_type = agb.types.ArrayType(
    'footer.border_line',
    (lambda project, context, parents: int(parents[-1]['border_height'])) 
)

border_array_pointer_type = agb.types.PointerType(
    'footer.border_array',
    (lambda project, context, parents: ('border', 2, False))
)

blocks_line_type = agb.types.ArrayType(
    'map_block',
    (lambda project, context, parents: int(parents[-2]['width']))
)

blocks_array_type = agb.types.ArrayType(
    'footer.blocks_line',
    (lambda project, context, parents: int(parents[-1]['height'])) 
)

blocks_array_pointer_type = agb.types.PointerType(
    'footer.blocks_array',
    (lambda project, context, parents: ('blocks', 2, False))
)

# Define a map footer type
footer_type = agb.types.Structure(
    [
        ('width', 'u32'),
        ('height', 'u32'),
        # The borders will be indexed [y][x]
        ('border', 'footer.border_array_pointer'),
        ('blocks', 'footer.blocks_array_pointer'),
        ('tileset_primary', 'tileset_pointer'),
        ('tileset_secondary', 'tileset_pointer'),
        ('border_width', 'u8'),
        ('border_height', 'u8'),
        ('field_1A', 'u16')
    ], 
    # Export the width and height of the blocks and border beforehand
    priorized_members=['width', 'height', 'border_width', 'border_height']
)

# These model declarations will be exported
default_model = {
    'map_block' : map_block_type,
    'footer.border_line' : border_line_type,
    'footer.border_array' : border_array_type,
    'footer.border_array_pointer' : border_array_pointer_type,
    'footer.blocks_line' : blocks_line_type,
    'footer.blocks_array' : blocks_array_type,
    'footer.blocks_array_pointer' : blocks_array_pointer_type,
    'footer' : footer_type
}