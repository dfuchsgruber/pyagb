import agb.types

# Define a type for map connections
connection_type = agb.types.Structure([
    ('direction', 'u32', 0),
    ('displacement', 's32', 0),
    ('bank', 'u8', 0),
    ('map_idx', 'u8', 0),
    ('field_A', 'u8', 0),
    ('field_B', 'u8', 0)
])

connection_array_type = agb.types.VariableSizeArrayType(
    'connection.connection',
    (1, ['connection_cnt'])
)

connection_array_pointer_type = agb.types.PointerType(
        'connection.connection_array',
        (lambda project, context, parents: ('connections', 2, False))
    )

# Define a type for map connection headers
connection_header_type = agb.types.Structure([
    ('connection_cnt', 'u32', 0),
    ('connections', 'connection.connection_array_pointer', 0)
])

# These model declarations will be exported
default_model = {
    'connection.connection' : connection_type,
    'connection.connection_array' : connection_array_type,
    'connection.connection_array_pointer' : connection_array_pointer_type,
    'connection.connection_header' : connection_header_type
}