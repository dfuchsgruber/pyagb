"""Data model for mapfooters."""

import agb.types
from agb.model import Model

map_dimension = agb.types.ScalarType('u32', default=1)
border_dimension = agb.types.ScalarType('u8', default=1)

# Define the map block bitfield type
map_block_type = agb.types.BitfieldType('u16', [
    ('block_idx', None, 10),
    ('level', None, 6)
])

border_line_type = agb.types.VariableSizeArrayType(
    'map_block',
    (2, ['border_width'])
)

border_array_type = agb.types.VariableSizeArrayType(
    'footer.border_line',
    (1, ['border_height'])
)

border_array_pointer_type = agb.types.PointerType(
    'footer.border_array',
    (lambda project, context, parents: ('border', 2, False))
)

blocks_line_type = agb.types.VariableSizeArrayType(
    'map_block',
    (2, ['width'])
)

blocks_array_type = agb.types.VariableSizeArrayType(
    'footer.blocks_line',
    (1, ['height'])
)

blocks_array_pointer_type = agb.types.PointerType(
    'footer.blocks_array',
    (lambda project, context, parents: ('blocks', 2, False))
)

# Define a map footer type
footer_type = agb.types.Structure(
    [
        # The map will be indexed [y][x]
        ('width', 'footer.map_dimension', 0),
        ('height', 'footer.map_dimension', 0),
        # The borders will be indexed [y][x]
        ('border', 'footer.border_array_pointer', 1),
        ('blocks', 'footer.blocks_array_pointer', 1),
        ('tileset_primary', 'tileset_pointer', 1),
        ('tileset_secondary', 'tileset_pointer', 1),
        ('border_width', 'footer.border_dimension', 0),
        ('border_height', 'footer.border_dimension', 0),
        ('field_1A', 'map_battle_style', 1)
    ],
    # Export the width and height of the blocks and border beforehand
    hidden_members=set([
        'border', 'blocks', 'tileset_primary', 'tileset_secondary',
        'width', 'height', 'border_width', 'border_height'
    ])
)

# These model declarations will be exported
default_model: Model = {
    'map_block' : map_block_type,
    'footer.border_line' : border_line_type,
    'footer.border_array' : border_array_type,
    'footer.border_array_pointer' : border_array_pointer_type,
    'footer.blocks_line' : blocks_line_type,
    'footer.blocks_array' : blocks_array_type,
    'footer.blocks_array_pointer' : blocks_array_pointer_type,
    'footer.map_dimension' : map_dimension,
    'footer.border_dimension' : border_dimension,
    'footer' : footer_type,
}
