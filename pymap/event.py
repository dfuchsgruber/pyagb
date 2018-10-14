import agb.types
from . import backend

person_type = agb.types.Structure([
    ('target_index', agb.types.u8),
    ('picture', agb.types.u8),
    ('field_2', agb.types.u8),
    ('field_3', agb.types.u8),
    ('x', agb.types.s16),
    ('y', agb.types.s16),
    ('level', agb.types.u8),
    ('behaviour', agb.types.ScalarType('u8', constant='person_behaviours')),
    ('behaviour_range', agb.types.u8),
    ('field_B', agb.types.u8),
    ('is_trainer', agb.types.u8),
    ('field_D', agb.types.u8),
    ('alter_radius', agb.types.u16),
    ('script', backend.OwScriptPointerType()),
    ('flag', agb.types.ScalarType('u16', constant='flags')),
    ('field_16', agb.types.u16)
])

trigger_type = agb.types.Structure([
    ('x', agb.types.s16),
    ('y', agb.types.s16),
    ('level', agb.types.u8),
    ('field_5', agb.types.u8),
    ('var', agb.types.ScalarType('u16', constant='vars')),
    ('value', agb.types.u16),
    ('field_A', agb.types.u8),
    ('field_B', agb.types.u8),
    ('script', backend.OwScriptPointerType())
])

warp_type = agb.types.Structure([
    ('x', agb.types.s16),
    ('y', agb.types.s16),
    ('level', agb.types.u8),
    ('target_idx', agb.types.u8),
    ('bank', agb.types.u8),
    ('map', agb.types.u8)    
])

def signpost_structure_get(parents):
    """ Returns the structure of a signpost based on its type field.
    
    Parameters:
    -----------
    parents : list
        The parents of the signpost.
    
    Returns:
    --------
    structure : 'item' or 'script'
        The structure of the signpost.
    """
    signpost_type = parents[-1]['type']
    if signpost_type < 5:
        return 'script'
    else:
        return 'item'

signpost_type = agb.types.Structure([
    ('x', agb.types.s16),
    ('y', agb.types.s16),
    ('level', agb.types.u8),
    ('type', agb.types.u8),
    ('field_6', agb.types.u8),
    ('field_7', agb.types.u8),
    ('value', agb.types.UnionType({
        'item' : agb.types.BitfieldType('u32', [
            ('item', 'items', 16),
            ('flag', None, 8),
            ('amount', None, 5),
            ('chunk', None, 3)
        ]),
        'script' : backend.OwScriptPointerType()
    }, signpost_structure_get))
])


event_header_type = agb.types.Structure([
    ('person_cnt', agb.types.u8),
    ('warp_cnt', agb.types.u8),
    ('trigger_cnt', agb.types.u8),
    ('signpost_cnt', agb.types.u8),
    ('persons', agb.types.PointerType(
        agb.types.ArrayType(person_type,
            # The size of the persons array is determined by the person_cnt
            (lambda parents: int(parents[-1]['person_cnt']))
        ),
        # The label is always persons, 2-aligned, and not global
        (lambda: ('persons', 2, False))
    )),
    ('warps', agb.types.PointerType(
        agb.types.ArrayType(warp_type,
            # The size of the warps array is determined by the warp_cnt
            (lambda parents: int(parents[-1]['warp_cnt']))
        ),
        # The label is always warps, 2-aligned, and not global
        (lambda: ('warps', 2, False))
    )),
    ('triggers', agb.types.PointerType(
        agb.types.ArrayType(trigger_type,
            # The size of the triggers array is determined by the trigger_cnt
            (lambda parents: int(parents[-1]['trigger_cnt']))
        ),
        # The label is always triggers, 2-aligned, and not global
        (lambda: ('triggers', 2, False))
    )),
    ('signposts', agb.types.PointerType(
        agb.types.ArrayType(signpost_type,
            # The size of the signposts array is determined by the signpost_cnt
            (lambda parents: int(parents[-1]['signpost_cnt']))
        ),
        # The label is always signposts, 2-aligned, and not global
        (lambda: ('signposts', 2, False))
    )),
])
