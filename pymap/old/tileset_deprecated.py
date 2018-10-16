import numpy as np
import json
import agb.palette
import agb.types

MAX_BLOCKS_PRIMARY = 0x200
MAX_BLOCKS_SECONDARY = 0x180
PALETTES_PRIMARY = 7
PALETTES_SECONDARY = 5

class Tileset:
    """ Map tileset class. """


    def __init__(self, is_primary, symbol, blocks=None, behaviours=None, palette=None, compressed_gfx=True):
        """ Initializes the map tileset.
        
        Parameters:
        -----------
        is_primary : bool
            If the tileset is a primary or not (=secondary) tileset.
        symbol : string
            The global symbol to export for this tileset.
        blocks : array-like, shape [num_blocks, 8] or None
            Block tilemap of the tileset.
            If None will be initialized with zeros.
        behaviours: array-like, shape [num_blocks, 8] or None
            Behaviours of the tileset.
            If None will be initialized with zeros.
        compressed_gfx : bool
            If the referrenced tileset gfx is lz77 compressed.
        """
        self.symbol = symbol
        self.is_primary = is_primary
        max_blocks = MAX_BLOCKS_PRIMARY if self.is_primary else MAX_BLOCKS_SECONDARY
        if blocks is None:
            self.blocks = np.zeros((max_blocks, 8), dtype=int)
        else:
            self.blocks = np.array(blocks, dtype=int)
            assert(self.blocks.shape[0] <= max_blocks and self.blocks.shape[1] == 8)
        if behaviours is None:
            self.behaviours = np.zeros((max_blocks, 8), dtype=int)
        else:
            self.behaviours = np.zeros((max_blocks, 8), dtype=int)
            assert(self.behaviours.shape[0] <= max_blocks and self.behaviours.shape[1] == 8)
        
        num_colors = 16 * (PALETTES_PRIMARY if self.is_primary else PALETTES_SECONDARY)
        if palette is None:
            self.palette = agb.palette.Palette(np.zeros((num_colors, 3), dtype=int))
        else:
            self.palette = palette
            assert(isinstance(palette, agb.palette.Palette))
            assert(len(palette) == num_colors)

        self.animation_initializer = '0'
        self.gfx = '0'
        self.compressed_gfx = compressed_gfx

    def to_assembly(self):
        """ Returns an assembly representation of the maptileset.
        
        Returns:
        --------
        as : string
            The assembly representation of the tileset
        """
        # Create tileset section
        ts = '\n'.join([
            f'.global {self.symbol}',
            '.align 2',
            f'{self.symbol}:',
            f'\t.byte {int(self.compressed_gfx)}',
            f'\t.byte {int(not self.is_primary)}',
            '\t.byte 0, 0',
            f'\t.word {self.gfx}',
            f'\t.word {self.symbol}_palette',
            f'\t.word {self.symbol}_blocks',
            f'\t.word {self.animation_initializer}'
            f'\t.word {self.symbol}_behaviours'
        ]) 

        # Create palette section
        pal = '\n'.join([
            f'.global {self.symbol}_palette',
            '.align 2',
            f'{self.symbol}_palette:',
            f'{",".join(map(str, self.palette.to_data()))}'
        ])

        # Create blocks section
        blocks = '\n'.join([
            f'.global {self.symbol}_blocks',
            '.align 2',
            f'{self.symbol}_blocks:',
            f'.hword {",".join(map(str, self.blocks.flatten()))}'
        ])

        # Create behaviours section
        behaviours = '\n'.join([
            f'.global {self.symbol}_behaviours',
            '.align 2',
            f'{self.symbol}_behaviours:',
            f'.word {",".join(map(behaviour_pack, self.behaviours))}'
        ])

        return '\n\n'.join([ts, pal, blocks, behaviours])

    def save(self, file_path):
        """ Saves this instance from a json representation.
        
        Parameters:
        -----------
        file_path : str
            The path to save this file at.
        """
        with open(file_path, 'w+') as f:
            f.write(json.dumps(
                {
                    'symbol' : self.symbol,
                    'is_primary' : self.is_primary,
                    'gfx' : self.gfx,
                    'animation_initializer' : self.animation_initializer,
                    'compressed_gfx' : self.compressed_gfx,
                    'blocks' : self.blocks.tolist(),
                    'behaviours' : self.behaviours.tolist(),
                    'palette' : self.palette.rgbs.tolist()
                }, indent='\t'
            ))


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
        dict = json.load(f)
    return Tileset(dict['is_primary'], dict['symbol'], blocks=np.array(dict['blocks']),
    behaviours=np.array(dict['behaviours']), palette=agb.palette.Palette(np.array(dict['palette'])),
    compressed_gfx=dict['compressed_gfx'])


# Behaviour bitfield assemble
behaviour_shifters = [(0, 8), (8, 13), (13, 17), (17, 23), (23, 26), (26, 28), (28, 31), (31, 32)]

def behaviour_pack(behaviour):
    """ Packs a behaviour into an assembly string.
    
    Parameters:
    -----------
    behaviour : iterable of length 8
        The behaviour values (may be ints or strings)
    
    Returns:
    --------
    packed : string
        The corresponding assembly string that assembles the behaviour value.
    """
    return '|'.join(f'({value} << {shift[0]})' for shift, value in zip(behaviour_shifters, behaviour))

def behaviour_unpack(value):
    """ Unpacks a behaviour integer (32-bit) into its eight bitfield.
    
    Parameters:
    -----------
    value : int (32-bit)
        The packed behaviour
    
    Returns:
    --------
    unpacked : ndarray, shape[8]
        The unpacked behaviour values
    """
    return np.fromiter(
        (value >> shift[0] & ((1 << (shift[1] - shift[0])) - 1)
        for shift in behaviour_shifters), int, count=8
    )

