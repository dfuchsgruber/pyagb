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
    ('alert_radius', agb.types.u16),
    ('script', backend.ow_script_pointer_type),
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
    ('script', backend.ow_script_pointer_type)
])

warp_type = agb.types.Structure([
    ('x', agb.types.s16),
    ('y', agb.types.s16),
    ('level', agb.types.u8),
    ('target_warp_idx', agb.types.u8),
    ('target_bank', agb.types.u8),
    ('target_map', agb.types.u8)    
])

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
        'script' : backend.ow_script_pointer_type
    }, backend.signpost_structure_get))
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
        (lambda parents: ('persons', 2, False))
    )),
    ('warps', agb.types.PointerType(
        agb.types.ArrayType(warp_type,
            # The size of the warps array is determined by the warp_cnt
            (lambda parents: int(parents[-1]['warp_cnt']))
        ),
        # The label is always warps, 2-aligned, and not global
        (lambda parents: ('warps', 2, False))
    )),
    ('triggers', agb.types.PointerType(
        agb.types.ArrayType(trigger_type,
            # The size of the triggers array is determined by the trigger_cnt
            (lambda parents: int(parents[-1]['trigger_cnt']))
        ),
        # The label is always triggers, 2-aligned, and not global
        (lambda parents: ('triggers', 2, False))
    )),
    ('signposts', agb.types.PointerType(
        agb.types.ArrayType(signpost_type,
            # The size of the signposts array is determined by the signpost_cnt
            (lambda parents: int(parents[-1]['signpost_cnt']))
        ),
        # The label is always signposts, 2-aligned, and not global
        (lambda parents: ('signposts', 2, False))
    )),
])
