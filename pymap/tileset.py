import numpy as np
import json
import agb.palette
import agb.types
from . import backend

MAX_BLOCKS_PRIMARY = 0x200
MAX_BLOCKS_SECONDARY = 0x180
PALETTES_PRIMARY = 7
PALETTES_SECONDARY = 5

class Tileset:
    """ Map tileset class. """


    def __init__(self, is_primary=True, initialize_arrays=True):
        """ Initializes the map tileset.
        
        Parameters:
        -----------
        is_primary : bool
            Wether the tileset is a primary tileset.
        initialize_arrays : bool
            If true, zeros will be allocated for blocks and behaviours.
        """
        # The first structural elements of the t ileset structure will be modelled
        # via the generic structure type. The references to the blocks and behaviours
        # will be modelled seperately.
        self.meta = agb.types.Structure([
            ('gfx_compressed', agb.types.ScalarType('u8'), 1),
            ('is_primary', agb.types.ScalarType('u8'), int(is_primary)),
            ('field_2', agb.types.ScalarType('u8'), 0),
            ('field_3', agb.types.ScalarType('u8'), 0),
            ('gfx', backend.GfxPointerType(compressed=True), 0)
        ])

        self.animation_initialize = '0'

        if initialize_arrays:
            # Initialize blocks, behaviours, 
            max_blocks = MAX_BLOCKS_PRIMARY if is_primary else MAX_BLOCKS_SECONDARY
            self.blocks = np.zeros((max_blocks, 8), dtype=int)
            self.behaviours = np.zeros((max_blocks, 8), dtype=int)
            num_colors = 16 * (PALETTES_PRIMARY if is_primary else PALETTES_SECONDARY)
            self.palette = agb.palette.Palette(np.zeros((num_colors, 3), dtype=int))
        else:
            self.blocks = None
            self.behaviours = None
            self.palette = None

    def from_data(self, rom, offset, project, context):
        """ Initializes the tileset members from rom binary data.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to initialize the tileset from
        offset : int
            The offset to initialize the tileset from
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list of str
            The context in which the data got initialized

        Returns:
        --------
        self : Tileset
            Self-reference
        offset : int
            The offset after processing the tileset structure.
        """
        # Initialize metadata
        _, offset = self.meta.from_data(rom, offset, project, context)

        # Initialize the arrays
        
        # Colors are 16-bit fields
        color_type = agb.types.BitfieldType('u16', [
            ('red', None, 5),
            ('blue', None, 5),
            ('green', None, 5)
        ])
        num_colors = 16 * (PALETTES_PRIMARY if self.meta.is_primary else PALETTES_SECONDARY)
        palette_offset, offset = rom.pointer[offset], offset + 4
        print(palette_offset)
        self.palette = np.empty((num_colors, 3), dtype=int)
        for i in range(num_colors):
            self.palette[i,:], palette_offset = color_type.from_data(rom, palette_offset, project, context)

        # Blocks are two 4-tile maps (= eight u16 values)
        blocks_offset, offset = rom.pointer[offset], offset + 4
        num_blocks = MAX_BLOCKS_PRIMARY if self.meta.is_primary else MAX_BLOCKS_SECONDARY
        self.blocks = np.array(rom.u16[blocks_offset : blocks_offset + num_blocks * 8 * 2 : 2]).reshape((num_blocks, 8))

        self.animation_initialize, offset = rom.u32[offset], offset + 4

        # Behaviours are 32-bit fields
        behaviour_type = agb.types.BitfieldType('u32', [
            ('behaviour', None, 9),
            ('hm_usage', None, 5),
            ('field_2', None, 4),
            ('field_3', None, 6),
            ('encounter_type', None, 3),
            ('field_5', None, 2),
            ('field_6', None, 2),
            ('field_7', None, 1)
        ])
        behaviour_offset, offset = rom.pointer[offset], offset + 4
        self.behaviours = np.empty((num_blocks, len(behaviour_type.structure)), dtype=object)
        for i in range(num_blocks):
            self.behaviours[i,:], behaviour_offset = behaviour_type.from_data(rom, behaviour_offset, project, context)
        
        return self, offset

    def save(self, file_path):
        """ Saves this instance from a json representation.
        
        Parameters:
        -----------
        file_path : str
            The path to save this file at.
        """
        with open(file_path, 'w+') as f:
            json.dump(f, 
                {
                    'meta' : self.meta.to_dict(),
                    'animation_initialize' : self.animation_initialize,
                    'blocks' : self.blocks.tolist(),
                    'behaviours' : self.behaviours.tolist(),
                    'palette' : self.palette.tolist()
                }, indent='\t'
            )


def from_file(file_path):
    """ Initializes a tileset instance from a file.
    
    Parameters:
    -----------
    file_path : str
        The path to the tileset file.
    
    Returns:
    --------
    tileset : Tileset
        The tileset instance.
    """
    with open(file_path) as f:
        serialized = json.load(f)
    
    tileset = Tileset(initialize_arrays=False)
    tileset.meta.from_dict(serialized['meta'])
    tileset.animation_initialize = serialized['animation_initialize']
    
    # Load arrays
    tileset.blocks = np.array(serialized['blocks'])
    tileset.behaviours = np.array(serialized['behaviours'])
    tileset.palette = np.array(serialized['palette'])



