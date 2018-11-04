import json

# Define a default configuration for pymap
default_configuration = {
    'pymap2s' : {
        # Include directive for pymap assemblies
        'include' : {
            'directive' : '.include "{constant}.s"',
        },
        # Link pymap file types to datatypes
        'file_types' : {
            'header' : 'header.header',
            'footer' : 'footer.footer',
            'tileset_primary' : 'tileset.tileset_primary',
            'tileset_secondary' : 'tileset.tileset_secondary'
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
    
    

