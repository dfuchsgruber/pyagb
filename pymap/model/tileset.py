import agb.types
from . import backend

MAX_BLOCKS_PRIMARY = 0x200
MAX_BLOCKS_SECONDARY = 0x180
PALETTES_PRIMARY = 7
PALETTES_SECONDARY = 5

# Define a rgb color type
color_type = agb.types.BitfieldType('u16', [
    ('red', None, 5),
    ('blue', None, 5),
    ('green', None, 5)
])

# Define an array type to define a block tilemap
block_tilemap_type = agb.types.ArrayType(agb.types.u16, lambda _: 8)

# Define a behaviour type
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

# Define a tileset type
tileset_type = agb.types.Structure([
    ('gfx_compressed', agb.types.u8),
    ('is_primary', agb.types.u8),
    ('field_2', agb.types.u8),
    ('field_3', agb.types.u8),
    ('gfx', backend.BackendPointerType(
        (lambda rom, offset, project, context, parents: backend.gfx(
            rom, offset, project, context, parents, int(parents[-1]['gfx_compressed']) != 0
        ))
    )),
    # Palette is a pointer to an array of colors
    ('palette', agb.types.PointerType(
        agb.types.ArrayType(color_type,
        # The size of the array is determined by the is_primary attribute of the structure
        (lambda parents: (PALETTES_PRIMARY * 16 if int(parents[-1]['is_primary']) != 0 else PALETTES_SECONDARY * 16))
        ), (lambda parents: ('palette', 2, False))
    )),
    ('blocks', agb.types.PointerType(
        agb.types.ArrayType(block_tilemap_type,
            (lambda parents: (MAX_BLOCKS_PRIMARY if int(parents[-1]['is_primary']) != 0 else MAX_BLOCKS_SECONDARY))
        ), (lambda parents: ('blocks', 2, False))
    )),
    ('animation_initialize', agb.types.u32),
    ('behaviours', agb.types.PointerType(
        agb.types.ArrayType(behaviour_type,
            (lambda parents: (MAX_BLOCKS_PRIMARY if int(parents[-1]['is_primary']) != 0 else MAX_BLOCKS_SECONDARY)),
        ), (lambda parents: ('behaviours', 2, False))
    ))
])
