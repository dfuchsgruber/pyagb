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
        'indent' : '\t'
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
        
    }
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
    
    

