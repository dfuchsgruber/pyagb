import agb.types
from . import event


# Define the levelscript types



"""
typedef struct mapheader {
    mapfooter *footer;
    map_events *events;
    levelscript_head *levelscripts;
    map_connections *connections;
    u16 music;
    u16 map_index; //main table is used when map is loaded
    u8 name_bank;
    u8 flash;
    u8 weather;
    u8 type;
    u8 worldmap_shape_id; // Used to associate the map with a shape in the worldmap pattern of the namespace
    u8 field_19;
    u8 show_name;
    u8 battle_style;

} mapheader;
"""