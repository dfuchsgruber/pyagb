import agb.types
from . import event, connection, backend

header_type = agb.types.Structure([
    ('footer', backend.footer_pointer_type),
    ('events', agb.types.PointerType(
        event.event_header_type,
        (lambda parents: ('events', 2, False))
    )),
    ('levelscripts', backend.levelscript_header_type),
    ('connections', agb.types.PointerType(
        connection.connection_header_type,
        (lambda parents: ('connections', 2, False))
    )),
    ('music', agb.types.ScalarType('u16', constant='songs')),
    ('footer_idx', agb.types.u16),
    ('namespace', agb.types.ScalarType('u8', constant='map_namespaces')),
    ('flash_type', agb.types.ScalarType('u8', constant='map_flash_types')),
    ('weather', agb.types.ScalarType('u8', constant='map_weathers')),
    ('type', agb.types.ScalarType('u8', constant='map_types')),
    ('field_18', agb.types.u8),
    ('field_19', agb.types.u8),
    ('show_name', agb.types.ScalarType('u8', constant='map_show_name_types')),
    ('battle_style', agb.types.ScalarType('u8', constant='map_battle_styles'))
])