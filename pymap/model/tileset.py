import agb.types
from . import backend

MAX_BLOCKS_PRIMARY = 0x280
MAX_BLOCKS_SECONDARY = 0x180
PALETTES_PRIMARY = 7
PALETTES_SECONDARY = 6

color_type = agb.types.BitfieldType('u16', [
    ('red', None, 5),
    ('blue', None, 5),
    ('green', None, 5)
])

palette_type = agb.types.ArrayType(
    'color',
    (lambda parents: 16)
)

palette_array_primary_type = agb.types.ArrayType(
    'palette',
    (lambda _: PALETTES_PRIMARY)
)

palette_array_secondary_type = agb.types.ArrayType(
    'palette',
    (lambda _: PALETTES_SECONDARY)
)


palette_array_primary_pointer_type = agb.types.PointerType(
    'tileset.palette_array_primary',
    (lambda parents: ('palettes', 2, False))
)

palette_array_secondary_pointer_type = agb.types.PointerType(
    'tileset.palette_array_secondary',
    (lambda parents: ('palettes', 2, False))
)

block_type = agb.types.BitfieldType(
    'u16', [
        ('tile_idx', None, 10),
        ('horizontal_flip', None, 1),
        ('vertical_flip', None, 1),
        ('palette_idx', None, 4)
    ]
)

block_tilemap_type = agb.types.ArrayType(
    'block',
    lambda _: 8
)

block_tilemap_array_primary_type = agb.types.ArrayType(
    'tileset.block_tilemap',
    (lambda _: MAX_BLOCKS_PRIMARY)
)

block_tilemap_array_secondary_type = agb.types.ArrayType(
    'tileset.block_tilemap',
    (lambda _: MAX_BLOCKS_SECONDARY)
)

block_tilemap_array_primary_pointer_type = agb.types.PointerType(
    'tileset.block_tilemap_array_primary',
    (lambda parents: ('blocks', 2, False))
)

block_tilemap_array_secondary_pointer_type = agb.types.PointerType(
    'tileset.block_tilemap_array_secondary',
    (lambda parents: ('blocks', 2, False))
)

behaviour_type = agb.types.BitfieldType('u32', [
    ('behaviour', None, 9),
    ('hm_usage', None, 5),
    ('field_2', None, 4),
    ('field_3', None, 6),
    ('encounter_type', None, 3),
    ('field_5', None, 2),
    ('field_6', None, 2),
    ('field_7', None, 1)
])

behaviour_array_primary_type = agb.types.ArrayType(
    'tileset.behaviour',
    (lambda _: MAX_BLOCKS_PRIMARY)
)

behaviour_array_secondary_type = agb.types.ArrayType(
    'tileset.behaviour',
    (lambda _: MAX_BLOCKS_SECONDARY)
)

behaviour_array_primary_pointer_type = agb.types.PointerType(
    'tileset.behaviour_array_primary',
    (lambda parents: ('behaviours', 2, False))
)

behaviour_array_secondary_pointer_type = agb.types.PointerType(
    'tileset.behaviour_array_secondary',
    (lambda parents: ('behaviours', 2, False))
)

gfx_pointer_type = backend.BackendPointerType(
    (lambda rom, offset, project, context, parents: backend.gfx(
        rom, offset, project, context, parents, int(parents[-1]['gfx_compressed']) != 0
    ))
)

# Define a primary tileset type
tileset_primary_type = agb.types.Structure([
    ('gfx_compressed', 'u8'),
    ('palette_displaced', 'u8'),
    ('field_2', 'u8'),
    ('field_3', 'u8'),
    ('gfx', 'tileset.gfx_pointer'),
    ('palettes', 'tileset.palette_array_primary_pointer'),
    ('blocks', 'tileset.block_tilemap_array_primary_pointer'),
    ('animation_initialize', 'u32'),
    ('behaviours', 'tileset.behaviour_array_primary_pointer')
])

# Define a secondary tileset type
tileset_secondary_type = agb.types.Structure([
    ('gfx_compressed', 'u8'),
    ('palette_displaced', 'u8'),
    ('field_2', 'u8'),
    ('field_3', 'u8'),
    ('gfx', 'tileset.gfx_pointer'),
    ('palettes', 'tileset.palette_array_secondary_pointer'),
    ('blocks', 'tileset.block_tilemap_array_secondary_pointer'),
    ('animation_initialize', 'u32'),
    ('behaviours', 'tileset.behaviour_array_secondary_pointer')
])


# These model declarations will be exported

default_model = {
    'color' : color_type,
    'palette' : palette_type,
    'tileset.palette_array_primary' : palette_array_primary_type,
    'tileset.palette_array_secondary' : palette_array_secondary_type,
    'tileset.palette_array_primary_pointer' : palette_array_primary_pointer_type,
    'tileset.palette_array_secondary_pointer' : palette_array_secondary_pointer_type,
    'block' : block_type,
    'tileset.block_tilemap' : block_tilemap_type,
    'tileset.block_tilemap_array_primary' : block_tilemap_array_primary_type,
    'tileset.block_tilemap_array_secondary' : block_tilemap_array_secondary_type,
    'tileset.block_tilemap_array_primary_pointer' : block_tilemap_array_primary_pointer_type,
    'tileset.block_tilemap_array_secondary_pointer' : block_tilemap_array_secondary_pointer_type,
    'tileset.behaviour' : behaviour_type,
    'tileset.behaviour_array_primary' : behaviour_array_primary_type,
    'tileset.behaviour_array_secondary' : behaviour_array_secondary_type,
    'tileset.behaviour_array_primary_pointer' : behaviour_array_primary_pointer_type,
    'tileset.behaviour_array_secondary_pointer' : behaviour_array_secondary_pointer_type,
    'tileset.tileset_primary' : tileset_primary_type,
    'tileset.tileset_secondary' : tileset_secondary_type,
    'tileset.gfx_pointer' : gfx_pointer_type
}