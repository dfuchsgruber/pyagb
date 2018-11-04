import agb.types

event_header_pointer_type = agb.types.PointerType(
    'event.event_header',
    (lambda project, context, parents: ('events', 2, False))
)

connection_header_pointer_type = agb.types.PointerType(
    'connection.connection_header',
    (lambda project, context, parents: ('connection_header', 2, False))
)

header_type = agb.types.Structure([
    ('footer', 'footer_pointer'),
    ('events', 'header.event_header_pointer'),
    ('levelscripts', 'levelscript_header_pointer'),
    ('connections', 'header.connection_header_pointer'),
    ('music', 'u16'),
    ('footer_idx', 'u16'),
    ('namespace', 'u8'),
    ('flash_type', 'u8'),
    ('weather', 'u8'),
    ('type', 'u8'),
    ('field_18', 'u8'),
    ('show_name', 'u8'),
    ('field_1A', 'u8'),
    ('battle_style', 'u8')
])

# These model declarations will be exported

default_model = {
    'header.event_header_pointer' : event_header_pointer_type,
    'header.connection_header_pointer' : connection_header_pointer_type,
    'header' : header_type
}