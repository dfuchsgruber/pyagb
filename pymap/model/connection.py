import agb.types

# Define a type for map connections
connection_type = agb.types.Structure([
    ('direction', agb.types.ScalarType('u32', constant='map_connections')),
    ('displacement', agb.types.s32),
    ('bank', agb.types.u8),
    ('map_idx', agb.types.u8),
    ('field_A', agb.types.u8),
    ('field_B', agb.types.u8)
])

# Define a type for map connection headers
connection_header_type = agb.types.Structure([
    ('connection_cnt', agb.types.u32),
    ('connections', agb.types.PointerType(
        agb.types.ArrayType(
            connection_type,
            (lambda parents: int(parents[-1]['connection_cnt']))
        ),
        (lambda parents: ('connections', 2, False))
    ))
])