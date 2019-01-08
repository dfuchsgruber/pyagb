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
            'buffers' : [0xFD, 0xFC],
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
        },
        'tileset_secondary' : {
            # Define how to access a certain palette given a tileset. If ['foo', 'bar'] is given
            # Then tileset['foo']['bar'] is expected to yield an array of palettes that can be indexed
            # with an integer
            'palettes_path' : ['palettes'],
            'gfx_path' : ['gfx'],
            'blocks_path' : ['blocks'],
            'datatype' : 'tileset_secondary',
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
        },
        'header' : {
            # Define a constant name for the namespaces or None if arbitrary strings should be allowed
            'namespace_constants' : None,
            # Define the datatype of the map header
            'datatype' : 'header',
            'namespace_path' : ['namespace'],
            'footer_path' : ['footer'],
            'footer_idx_path' : ['footer_idx'],
        },
        'display' : {
            # Show how many border blocks will be padded to the map display (x, y)
            'border_padding' : [7, 5],
            # R,G,B,Alpha value of the borders 
            'border_color' : [0, 0, 0, 0.3],
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
    
    

