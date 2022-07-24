import json

# Define a default configuration for pymap
default_configuration = {
    'pymap2s' : {
        # Include directive for pymap assemblies
        'include' : {
            'directive' : '.include "{constant}.s"',
        }
    },
    # Define additional models that may override default models
    'model' : [],
    'json' : {
        # Define the indent for outputting json files
        'indent' : '\t',
        # Define the default encoding for json files
        'encoding' : 'utf-8'
    },
    'rom' : {
        # The offset where the rom will be loaded into RAM
        'offset' : 0x08000000
    },
    'string' : {
        # Define a character mapping. If None, no encoder / decoder can be used.
        'charmap' : None,
        # A sequence that terminates strings.
        'tail' : [
            0xFF
        ],
        'as' : {
            # Directive for string assemblies
            'directives' : {
                # Compile a string plainly into bytes 
                # Example:
                # .string "..."
                'std' : '.string',
                # Compile a string and break it automatically to fit a box
                # Example:
                # .autostring WIDTH HEIGHT "..."
                'auto' : '.autostring',
                # Compile a string and pad it to a certain length in bytes
                # Example
                # .stringpad SIZE "..."
                'padded' : '.stringpad'
            }
        },
        'c' : {
            # Macro to enclose c strings
            'macro' : 'PSTRING'
        },
        # Define special characters such as newline, etc.
        'characters' : {
            'newline' : 0xFE,
            'scroll' : 0xFA,
            'paragraph' : 0xFB,
            'buffers' : [0xFD],
            'control_codes' : {
                0xFC : {
                    0x8 : 1, # Text delay
                    0xB : 2, # Play Song
                    0xC : 1, # Tall plus
                    0x10 : 2, # Play Sound Effect
                },
            },
            'delimiters' : [0x0],
            'max_buffer_size' : 10
        }  
    },
    'pymap' : {
        # Configure how to handle a tileset dictionary structure
        'tileset_primary' : {
            # Define how to access a certain palette given a tileset. If ['foo', 'bar'] is given
            # Then tileset['foo']['bar'] is expected to yield an array of palettes that can be indexed
            # with an integer
            'palettes_path' : ['palettes'],
            'gfx_path' : ['gfx'],
            'blocks_path' : ['blocks'],
            'datatype' : 'tileset_primary',
            'behaviours_path' : ['behaviours'],
            'behaviour_datatype' : 'tileset.behaviour',
            'gfx_compressed_path' : ['gfx_compressed'],
            'animation_path' : ['animation_initialize'],
        },
        'tileset_secondary' : {
            # Define how to access a certain palette given a tileset. If ['foo', 'bar'] is given
            # Then tileset['foo']['bar'] is expected to yield an array of palettes that can be indexed
            # with an integer
            'palettes_path' : ['palettes'],
            'gfx_path' : ['gfx'],
            'blocks_path' : ['blocks'],
            'datatype' : 'tileset_secondary',
            'behaviours_path' : ['behaviours'],
            'behaviour_datatype' : 'tileset.behaviour',
            'gfx_compressed_path' : ['gfx_compressed'],
            'animation_path' : ['animation_initialize'],
        },
        'footer' : {
            'map_width_path' : ['width'],
            'map_height_path' : ['height'],
            'map_blocks_path' : ['blocks'],
            'border_width_path' : ['border_width'],
            'border_height_path' : ['border_height'],
            'border_path' : ['border'],
            'tileset_primary_path' : ['tileset_primary'],
            'tileset_secondary_path' : ['tileset_secondary'],
            # Define the datatype of the map footer
            'datatype' : 'footer',
            # Define max width of the map
            'map_width_max' : 127,
            'map_height_max' : 127,
            'border_width_max' : 7,
            'border_height_max' : 5,
        },
        'header' : {
            # Define a constant name for the namespaces or None if arbitrary strings should be allowed
            'namespace_constants' : None,
            # Define the datatype of the map header
            'datatype' : 'header',
            'namespace_path' : ['namespace'],
            'footer_path' : ['footer'],
            'footer_idx_path' : ['footer_idx'],
            # Define different event types
            'events' : [
                {
                    'name' : 'Person',
                    'datatype' : 'event.person',
                    'size_path' : ['events', 'person_cnt'],
                    'events_path' : ['events', 'persons'],
                    'box_color' : [0.4, 0.7, 0.4, 0.7],
                    'text_color' : [1.0, 1.0, 1.0, 0.7],
                    # Define the path for the position of an event
                    'x_path' : ['x'], 'y_path' : ['y'],
                },
                {
                    'name' : 'Warp',
                    'datatype' : 'event.warp',
                    'size_path' : ['events', 'warp_cnt'],
                    'events_path' : ['events', 'warps'],
                    'box_color' : [0.6, 0.35, 0.85, 0.7],
                    'text_color' : [1.0, 1.0, 1.0, 0.7],
                    # Define the path for the position of an event
                    'x_path' : ['x'], 'y_path' : ['y'],
                    # Warps enable a button in the widget to warp to a map
                    'goto_header_button_button_enabled' : True,
                    'target_bank_path' : ['target_bank'],
                    'target_map_idx_path' : ['target_map'],
                    'target_warp_idx_path' : ['target_warp_idx'],
                },
                {
                    'name' : 'Sign',
                    'datatype' : 'event.signpost',
                    'size_path' : ['events', 'signpost_cnt'],
                    'events_path' : ['events', 'signposts'],
                    'box_color' : [0.8, 0.35, 0.3, 0.7],
                    'text_color' : [1.0, 1.0, 1.0, 0.7],
                    # Define the path for the position of an event
                    'x_path' : ['x'], 'y_path' : ['y'],
                },
                {
                    'name' : 'Trigger',
                    'datatype' : 'event.trigger',
                    'size_path' : ['events', 'trigger_cnt'],
                    'events_path' : ['events', 'triggers'],
                    'box_color' : [0.3, 0.45, 0.75, 0.7],
                    'text_color' : [1.0, 1.0, 1.0, 0.7],
                    # Define the path for the position of an event
                    'x_path' : ['x'], 'y_path' : ['y'],
                }
            ],
            'connections' : {
                'connections_path' : ['connections', 'connections'],
                'connections_size_path' : ['connections', 'connection_cnt'],
                # Define all type of recognized connection values that are to represented visually on the border
                'connection_types' : {
                    1 : 'south',
                    2 : 'north',
                    3 : 'west',
                    4 : 'east',
                },
                'connection_type_path' : ['direction'],
                'connection_offset_path' : ['displacement'],
                'connection_bank_path' : ['bank'],
                'connection_map_idx_path' : ['map_idx'],
                'datatype' : 'connection.connection' 
            }
            
        },
        'project' : {
            # Automatically save project on changes to labels or added resources. This prevents inconsistencies.
            'autosave' : True
        },
        'display' : {
            # Show how many border blocks will be padded to the map display (x, y)
            'border_padding' : [7, 5],
            # R,G,B,Alpha value of the borders 
            'border_color' : [0.0, 0.0, 0.0, 0.4],
            # Define a python script that provides a function to associate events with an Pilow image
            # The event image backend should contain a function:
            # def get_event_to_image() that returns an object that can provide images from events.
            # This object must have a method that fulfils the following interface:
            # def event_to_image(self, event, event_type, project)
            # that either returns None if no association was found or a triplet (PilImage, horizontal_displacement, vertical_displacement)
            # that indicates which image to use and how it is displaced w.r.t. to the upper left corner of its block
            'event_to_image_backend' : None,
            # R,G,B,Alpha value of the connection maps 
            'connection_color' : [1.0, 1.0, 1.0, 0.2],
            # R,G,B,Alpha value of the currently active connection map
            'connection_active_color' : [1.0, 0.2, 0.2, 0.15],
            # R,G,B,Alpha value of the connections maps border boxes
            'connection_border_color' : [1.0, 1.0, 1.0, 1.0],
            # R,G,B,Alpha value of the currently active connections map border boxes
            'connection_active_border_color' : [1.0, 0.0, 0.0, 1.0],
        }
    },
}

# Helper function to recursively iterate over the dicts
def update_dict(src, target):
    """ Updates the values in the target dictionary with all values
    from the src dictionary recursively.
    
    Parameters:
    -----------
    src : dict
        The dictionary that contains the values that will be used to
        update.
    target : dict
        The dictionary that contains the default values and will be
        extended / overwritten by the values of src.
    """
    for key, value in src.items():
        if isinstance(value, dict):
            if not key in target:
                target[key] = {}
            update_dict(value, target[key])
        else:
            target[key] = value

def get_configuration(file_path):
    """ Creates a configuration with default values overwritten by
    the custom .config file. """
    configuration = default_configuration.copy()
    with open(file_path) as f:
        custom_configuration = json.load(f)
    update_dict(custom_configuration, configuration)
    return configuration
    
    

