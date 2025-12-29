"""Configuration for pymap."""

import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence, TypeAlias, TypedDict

from agb.model.type import ModelContext, ModelValue
from pymap.gui.types import ConnectionType

AttributePathType: TypeAlias = Sequence[str | int]


class Pymap2sIncludeConfigType(TypedDict):
    """Configuration for pymap2s include directive."""

    directive: str


class Pymap2sConfigType(TypedDict):
    """Configuration for pymap2s assembler."""

    include: Pymap2sIncludeConfigType


class JsonConfigType(TypedDict):
    """Configuration for json files."""

    indent: str
    encoding: str


class RomConfigType(TypedDict):
    """Configuration for rom files."""

    offset: int


class StringAsDirectivesConfigType(TypedDict):
    """Configuration for string assembly directives."""

    std: str
    auto: str
    padded: str


class StringAsConfigType(TypedDict):
    """Configuration for string assembly."""

    directives: StringAsDirectivesConfigType
    comment_delimiter: str


class StringCConfigType(TypedDict):
    """Configuration for string C macro."""

    macro: str


# for each parameter, how many bytes it consumes
StringCharactersControlCodeType: TypeAlias = dict[int, int]


class StringCharactersConfigType(TypedDict):
    """Configuration for string characters."""

    newline: int
    scroll: int
    paragraph: int
    buffers: list[int]
    control_codes: dict[int, StringCharactersControlCodeType]
    delimiters: list[int]
    max_buffer_size: int


StringConfigType = TypedDict(
    'StringConfigType',
    {
        'charmap': None | str,
        'tail': tuple[int],
        'as': StringAsConfigType,
        'c': StringCConfigType,
        'characters': StringCharactersConfigType,
    },
)


class PymapTilesetPrimaryConfigType(TypedDict):
    """Configuration for primary tileset in pymap."""

    palettes_path: ModelContext
    gfx_path: AttributePathType
    blocks_path: ModelContext
    block_datatype: str
    datatype: str
    behaviours_path: ModelContext
    behaviour_datatype: str
    gfx_compressed_path: AttributePathType
    animation_path: ModelContext


class PymapTilesetSecondaryConfigType(TypedDict):
    """Configuration for secondary tileset in pymap."""

    palettes_path: ModelContext
    gfx_path: AttributePathType
    blocks_path: ModelContext
    block_datatype: str
    datatype: str
    behaviours_path: ModelContext
    behaviour_datatype: str
    gfx_compressed_path: AttributePathType
    animation_path: ModelContext


class PymapFooterConfigType(TypedDict):
    """Configuration for pymap footer."""

    map_width_path: ModelContext
    map_height_path: ModelContext
    map_blocks_path: ModelContext
    border_width_path: ModelContext
    border_height_path: ModelContext
    border_path: ModelContext
    tileset_primary_path: AttributePathType
    tileset_secondary_path: AttributePathType
    datatype: str
    map_width_max: int
    map_height_max: int
    border_width_max: int
    border_height_max: int


class PymapEventTemplateConfigType(TypedDict):
    """Configuration for templates for an event type."""


EventTemplateType: TypeAlias = Sequence[tuple[ModelContext, ModelValue]]


class PymapEventConfigType(TypedDict):
    """Configuration for an event type."""

    datatype: str
    display_letter: str
    size_path: ModelContext
    events_path: ModelContext
    box_color: list[float]
    text_color: list[float]
    x_path: ModelContext
    y_path: ModelContext
    templates: dict[str, EventTemplateType]


class PymapEventPersonConfigType(PymapEventConfigType):
    """Person event type configuration."""


class PymapEventWarpConfigType(PymapEventConfigType):
    """Warp event type configuration."""

    goto_header_button_button_enabled: bool
    target_bank_path: list[str]
    target_map_idx_path: list[str]
    target_warp_idx_path: list[str]


class PymapEventSignConfigType(PymapEventConfigType):
    """Signpost event type configuration."""


class PymapEventTriggerConfigType(PymapEventConfigType):
    """Trigger event type configuration."""


class PymapConnectionConnectionConfigType(TypedDict):
    """Configuration for connections in pymap."""

    connections_path: ModelContext
    connections_size_path: ModelContext
    connection_types: dict[int | str, str]
    connection_type_path: ModelContext
    connection_offset_path: ModelContext
    connection_bank_path: ModelContext
    connection_map_idx_path: ModelContext
    datatype: str


class PymapHeaderConfigType(TypedDict):
    """Configuration for pymap header."""

    namespace_constants: None | str
    datatype: str
    namespace_path: AttributePathType
    footer_path: AttributePathType
    footer_idx_path: list[str | int]
    events: dict[str, PymapEventConfigType]
    connections: PymapConnectionConnectionConfigType


class PymapProjectConfigType(TypedDict):
    """Configuration for pymap project."""

    autosave: bool


class PymapColorType(NamedTuple):
    """RGBA color type for pymap."""

    r: float
    g: float
    b: float
    a: float


class PymapDisplayConfigType(TypedDict):
    """Configuration for pymap display."""

    border_padding: list[int]
    border_color: list[float]
    connection_color: PymapColorType
    connection_active_color: PymapColorType
    connection_border_color: PymapColorType
    connection_active_border_color: PymapColorType
    smart_shape_blocks_per_row: int


class PymapConfigType(TypedDict):
    """Configuration for pymap."""

    tileset_primary: PymapTilesetPrimaryConfigType
    tileset_secondary: PymapTilesetSecondaryConfigType
    footer: PymapFooterConfigType
    header: PymapHeaderConfigType
    project: PymapProjectConfigType
    display: PymapDisplayConfigType


class ConfigType(TypedDict):
    """Configuration for pymap."""

    pymap2s: Pymap2sConfigType
    model: list[str]
    json: JsonConfigType
    rom: RomConfigType
    string: StringConfigType
    pymap: PymapConfigType
    backend: None | str


# Define a default configuration for pymap
default_configuration = ConfigType(
    {
        'pymap2s': Pymap2sConfigType(
            # Include directive for pymap assemblies
            include=Pymap2sIncludeConfigType(directive='.include "{constant}.s"')
        ),
        # Define additional models that may override default models
        'model': [],
        'json': JsonConfigType(
            # Define the indent for outputting json files
            indent='\t',
            # Define the default encoding for json files
            encoding='utf-8',
        ),
        'rom': RomConfigType(
            # The offset where the rom will be loaded into RAM
            offset=0x08000000
        ),
        'string': StringConfigType(
            {
                # Define a character mapping. If None, no encoder / decoder can be used.
                'charmap': None,
                # A sequence that terminates strings.
                'tail': (0xFF,),
                'as': StringAsConfigType(
                    # Directive for string assemblies
                    directives=StringAsDirectivesConfigType(
                        # Compile a string plainly into bytes
                        # Example:
                        # .string "..."
                        std='.string',
                        # Compile a string and break it automatically to fit a box
                        # Example:
                        # .autostring WIDTH HEIGHT "..."
                        auto='.autostring',
                        # Compile a string and pad it to a certain length in bytes
                        # Example
                        # .stringpad SIZE "..."
                        padded='.stringpad',
                    ),
                    comment_delimiter='@',
                ),
                'c': StringCConfigType(
                    # Macro to enclose c strings
                    macro='PSTRING'
                ),
                # Define special characters such as newline, etc.
                'characters': StringCharactersConfigType(
                    newline=0xFE,
                    scroll=0xFA,
                    paragraph=0xFB,
                    buffers=[0xFD],
                    control_codes={
                        0xFC: {
                            0x8: 1,  # Text delay
                            0xB: 2,  # Play Song
                            0xC: 1,  # Tall plus
                            0x10: 2,  # Play Sound Effect
                        },
                    },
                    delimiters=[0x0],
                    max_buffer_size=10,
                ),
            }
        ),
        'pymap': PymapConfigType(
            {
                # Configure how to handle a tileset dictionary structure
                'tileset_primary': PymapTilesetPrimaryConfigType(
                    {
                        # Define how to access a certain palette given a tileset.
                        # If ['foo', 'bar'] is given
                        # Then tileset['foo']['bar'] is expected to yield an array
                        # of palettes that can be indexed
                        # with an integer
                        'palettes_path': ['palettes'],
                        'gfx_path': ['gfx'],
                        'blocks_path': ['blocks'],
                        'block_datatype': 'block',
                        'datatype': 'tileset_primary',
                        'behaviours_path': ['behaviours'],
                        'behaviour_datatype': 'tileset.behaviour',
                        'gfx_compressed_path': ['gfx_compressed'],
                        'animation_path': ['animation_initialize'],
                    }
                ),
                'tileset_secondary': PymapTilesetSecondaryConfigType(
                    {
                        # Define how to access a certain palette given a tileset.
                        # If ['foo', 'bar'] is given
                        # Then tileset['foo']['bar'] is expected to yield an array
                        # of palettes that can be indexed
                        # with an integer
                        'palettes_path': ['palettes'],
                        'gfx_path': ['gfx'],
                        'blocks_path': ['blocks'],
                        'block_datatype': 'block',
                        'datatype': 'tileset_secondary',
                        'behaviours_path': ['behaviours'],
                        'behaviour_datatype': 'tileset.behaviour',
                        'gfx_compressed_path': ['gfx_compressed'],
                        'animation_path': ['animation_initialize'],
                    }
                ),
                'footer': PymapFooterConfigType(
                    {
                        'map_width_path': ['width'],
                        'map_height_path': ['height'],
                        'map_blocks_path': ['blocks'],
                        'border_width_path': ['border_width'],
                        'border_height_path': ['border_height'],
                        'border_path': ['border'],
                        'tileset_primary_path': ['tileset_primary'],
                        'tileset_secondary_path': ['tileset_secondary'],
                        # Define the datatype of the map footer
                        'datatype': 'footer',
                        # Define max width of the map
                        'map_width_max': 127,
                        'map_height_max': 127,
                        'border_width_max': 7,
                        'border_height_max': 5,
                    }
                ),
                'header': PymapHeaderConfigType(
                    {
                        # Define a constant name for the namespaces or None if arbitrary
                        # strings should be allowed
                        'namespace_constants': None,
                        # Define the datatype of the map header
                        'datatype': 'header',
                        'namespace_path': ['namespace'],
                        'footer_path': ['footer'],
                        'footer_idx_path': ['footer_idx'],
                        # Define different event types
                        'events': {
                            'Person': PymapEventPersonConfigType(
                                {
                                    'datatype': 'event.person',
                                    'size_path': ['events', 'person_cnt'],
                                    'events_path': ['events', 'persons'],
                                    'box_color': [0.4, 0.7, 0.4, 0.7],
                                    'text_color': [1.0, 1.0, 1.0, 0.7],
                                    # Define the path for the position of an event
                                    'x_path': ['x'],
                                    'y_path': ['y'],
                                    'templates': {},
                                    'display_letter': 'P',
                                }
                            ),
                            'Warp': PymapEventWarpConfigType(
                                {
                                    'datatype': 'event.warp',
                                    'size_path': ['events', 'warp_cnt'],
                                    'events_path': ['events', 'warps'],
                                    'box_color': [0.6, 0.35, 0.85, 0.7],
                                    'text_color': [1.0, 1.0, 1.0, 0.7],
                                    # Define the path for the position of an event
                                    'x_path': ['x'],
                                    'y_path': ['y'],
                                    # Warps enable a button in the widget to warp to a
                                    # map
                                    'goto_header_button_button_enabled': True,
                                    'target_bank_path': ['target_bank'],
                                    'target_map_idx_path': ['target_map'],
                                    'target_warp_idx_path': ['target_warp_idx'],
                                    'templates': {},
                                    'display_letter': 'W',
                                }
                            ),
                            'Sign': PymapEventSignConfigType(
                                {
                                    'datatype': 'event.signpost',
                                    'size_path': ['events', 'signpost_cnt'],
                                    'events_path': ['events', 'signposts'],
                                    'box_color': [0.8, 0.35, 0.3, 0.7],
                                    'text_color': [1.0, 1.0, 1.0, 0.7],
                                    # Define the path for the position of an event
                                    'x_path': ['x'],
                                    'y_path': ['y'],
                                    'templates': {},
                                    'display_letter': 'S',
                                }
                            ),
                            'Trigger': PymapEventTriggerConfigType(
                                {
                                    'datatype': 'event.trigger',
                                    'size_path': ['events', 'trigger_cnt'],
                                    'events_path': ['events', 'triggers'],
                                    'box_color': [0.3, 0.45, 0.75, 0.7],
                                    'text_color': [1.0, 1.0, 1.0, 0.7],
                                    # Define the path for the position of an event
                                    'x_path': ['x'],
                                    'y_path': ['y'],
                                    'templates': {},
                                    'display_letter': 'T',
                                }
                            ),
                        },
                        'connections': PymapConnectionConnectionConfigType(
                            {
                                'connections_path': ['connections', 'connections'],
                                'connections_size_path': [
                                    'connections',
                                    'connection_cnt',
                                ],
                                # Define all type of recognized connection values that
                                # are to represented visually on the border
                                'connection_types': {
                                    1: ConnectionType.SOUTH,
                                    2: ConnectionType.NORTH,
                                    3: ConnectionType.WEST,
                                    4: ConnectionType.EAST,
                                },
                                'connection_type_path': ['direction'],
                                'connection_offset_path': ['displacement'],
                                'connection_bank_path': ['bank'],
                                'connection_map_idx_path': ['map_idx'],
                                'datatype': 'connection.connection',
                            }
                        ),
                    }
                ),
                'project': PymapProjectConfigType(
                    {
                        # Automatically save project on changes to labels or added
                        # resources. This prevents inconsistencies.
                        'autosave': True
                    }
                ),
                'display': PymapDisplayConfigType(
                    {
                        # Show how many border blocks will be padded to the map display
                        # (x, y)
                        'border_padding': [7, 5],
                        # R,G,B,Alpha value of the borders
                        'border_color': [0.0, 0.0, 0.0, 0.4],
                        # R,G,B,Alpha value of the connection maps
                        'connection_color': PymapColorType(1.0, 1.0, 1.0, 0.2),
                        # R,G,B,Alpha value of the currently active connection map
                        'connection_active_color': PymapColorType(1.0, 0.2, 0.2, 0.15),
                        # R,G,B,Alpha value of the connections maps border boxes
                        'connection_border_color': PymapColorType(1.0, 1.0, 1.0, 1.0),
                        # R,G,B,Alpha value of the currently active connections map
                        # border boxes
                        'connection_active_border_color': PymapColorType(
                            1.0, 0.0, 0.0, 1.0
                        ),
                        # How many blocks the pool for smart shapes has per row
                        'smart_shape_blocks_per_row': 4,
                    }
                ),
            }
        ),
        # Path to a project-specific backend.
        'backend': None,
    },
)


# Helper function to recursively iterate over the dicts
def update_dict(src: dict[Any, Any], target: dict[Any, Any]):
    """Recursively updates the values in the target dictionary.

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
            if key not in target:
                target[key] = {}
            update_dict(value, target[key])  # type: ignore
        else:
            target[key] = value


def get_configuration(file_path: str) -> ConfigType:
    """Creates a configuration with default values overwritten.

    Overrides are defined by the custom .config file.
    """
    configuration: ConfigType = default_configuration.copy()
    with open(Path(file_path)) as f:
        custom_configuration = json.load(f)
    update_dict(custom_configuration, configuration)  # type: ignore
    return configuration
