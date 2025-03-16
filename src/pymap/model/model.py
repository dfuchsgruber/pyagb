"""The default data model for pymap.

This model can be extended per project in its configuration file.
"""

from pathlib import Path
from typing import Mapping, TypeAlias

import agb.types
import pymap.model.backend
import pymap.model.connection
import pymap.model.event
import pymap.model.footer
import pymap.model.header
import pymap.model.tileset

Model: TypeAlias = Mapping[str, agb.types.Type]

# Basic scalar types
default_model: dict[str, agb.types.Type] = {
    'u8': agb.types.ScalarType('u8'),
    's8': agb.types.ScalarType('s8'),
    's16': agb.types.ScalarType('s16'),
    'u16': agb.types.ScalarType('u16'),
    'u32': agb.types.ScalarType('u32'),
    's32': agb.types.ScalarType('s32'),
    'int': agb.types.ScalarType('s32'),
    'pointer': agb.types.ScalarType('pointer'),
}

# Import models from other model files
default_models: list[Model] = [
    pymap.model.event.default_model,
    pymap.model.footer.default_model,
    pymap.model.connection.default_model,
    pymap.model.tileset.default_model,
    pymap.model.header.default_model,
    pymap.model.backend.default_model,
]

for model in default_models:
    default_model |= model


def get_model(models_file_paths: list[str]) -> Model:
    """Gets the models for pymap.

    Parameters:
    -----------
    model_files : list of str
        List of path to a python scripts that create new types for
        the data model. All new types must be available in
        a variable 'models' which is a dict that maps
        type names to their respective instances.

    Returns:
    --------
    model : Model
        Maps from strings to type classes.
    """
    model = default_model.copy()
    for models_file_path in models_file_paths:
        # Remove models from previous files
        if 'models_to_export' in locals():
            del locals()['models_to_export']
        with open(Path(models_file_path)) as f:
            exec(f.read())
            # Try to get models
            try:
                models = locals()['models_to_export']
            except Exception as e:
                raise RuntimeError(
                    f'{models_file_path} does not provide a '
                    f"'models_to_export' variable (Exception: {e})"
                )
            for key, value in models.items():
                model[key] = value
    return model
