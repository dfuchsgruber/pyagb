import agb.types

# Define a type for map connections
connection_type = agb.types.Structure([
    ('direction', 'u32'),
    ('displacement', 's32'),
    ('bank', 'u8'),
    ('map_idx', 'u8'),
    ('field_A', 'u8'),
    ('field_B', 'u8')
])

connection_array_type = agb.types.ArrayType(
    'connection.connection',
    (lambda project, context, parents: int(parents[-1]['connection_cnt']))
)

connection_array_pointer_type = agb.types.PointerType(
        'connection.connection_array',
        (lambda project, context, parents: ('connections', 2, False))
    )

# Define a type for map connection headers
connection_header_type = agb.types.Structure([
    ('connection_cnt', 'u32'),
    ('connections', 'connection.connection_array_pointer')
])

# These model declarations will be exported
default_model = {
    'connection.connection' : connection_type,
    'connection.connection_array' : connection_array_type,
    'connection.connection_array_pointer' : connection_array_pointer_type,
    'connection.connection_header' : connection_header_type
}