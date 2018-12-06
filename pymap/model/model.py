# Creates the default models for pymap
import agb.types
import pymap.model.event, pymap.model.footer, pymap.model.header, pymap.model.tileset, pymap.model.backend, pymap.model.connection

# Basic scalar types
default_model = {
    'u8' : agb.types.ScalarType('u8'),
    's8' : agb.types.ScalarType('s8'),
    's16' : agb.types.ScalarType('s16'),
    'u16' : agb.types.ScalarType('u16'),
    'u32' : agb.types.ScalarType('u32'),
    's32' : agb.types.ScalarType('s32'),
    'int' : agb.types.ScalarType('s32'),
    'pointer' : agb.types.ScalarType('pointer'),
}

# Import models from other model files
default_models = [
    pymap.model.event.default_model,
    pymap.model.footer.default_model,
    pymap.model.connection.default_model,
    pymap.model.tileset.default_model,
    pymap.model.header.default_model,
    pymap.model.backend.default_model
]

for model in default_models:
    for typename, datatype in model.items():
        default_model[typename] = datatype

def get_model(models_file_paths):
    """ Gets the models for pymap.
    
    Parameters:
    -----------
    model_files : list of str
        List of path to a python scripts that create new types for
        the data model. All new types must be available in
        a variable 'models' which is a dict that maps
        type names to their respective instances.

    Returns:
    --------
    model : dict
        Maps from strings to type classes.
    """
    model = default_model.copy()
    for models_file_path in models_file_paths:
        # Remove models from previous files
        if 'models_to_export' in locals():
            del locals()['models_to_export']
        with open(models_file_path) as f:
            exec(f.read())
            # Try to get models
            try:
                models = locals()['models_to_export']
            except:
                raise RuntimeError(f'{models_file_path} does not provide a \'models_to_export\' variable')
            for key, value in models.items():
                model[key] = value
    return model 