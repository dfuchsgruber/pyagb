"""Datamodel for map headers."""

import agb.types
from agb.model import Model

event_header_pointer_type = agb.types.PointerType(
    'event.event_header',
    (lambda project, context, parents: ('events', 2, False))
)

connection_header_pointer_type = agb.types.PointerType(
    'connection.connection_header',
    (lambda project, context, parents: ('connection_header', 2, False))
)

header_type = agb.types.Structure([
    ('footer', 'footer_pointer', 0),
    ('events', 'header.event_header_pointer', 0),
    ('levelscripts', 'levelscript_header_pointer', 0),
    ('connections', 'header.connection_header_pointer', 0),
    ('music', 'u16', 0),
    ('footer_idx', 'u16', 0),
    ('namespace', 'u8', 0),
    ('flash_type', 'u8', 0),
    ('weather', 'u8', 0),
    ('type', 'u8', 0),
    ('field_18', 'u8', 0),
    ('show_name', 'u8', 0),
    ('field_1A', 'u8', 0),
    ('battle_style', 'u8', 0)
], hidden_members=set([
    'events', 'connections', 'footer', 'footer_idx', 'namespace'
])
)

# These model declarations will be exported

default_model: Model = {
    'header.event_header_pointer' : event_header_pointer_type,
    'header.connection_header_pointer' : connection_header_pointer_type,
    'header' : header_type
}
