import numpy as np

import PIL.Image
import struct

class Palette:

    def __init__(self, rgbs):
        """ Initalizes the palette structure. 
        
        Parameters:
        -----------
        rgbs : array-like
            Sequence of triplets (Red, Green, Blue)
        """
        self.rgbs = np.array(rgbs, dtype=int)

    def __getitem__(self, key):
        return Palette(self.rgbs[key])

    def __setitem__(self, key, item):
        if isinstance(item, Palette):
            item = item.rgbs
        self.rgbs[key] = item

    def __len__(self):
        return self.rgbs.shape[0]

    def to_pil_palette(self):
        """ Creates a palette that can be used for Pilow images.
        
        Returns:
        --------
        palette : ndarray, shape[len(self) * 3]
            Flattened array where each group of three elements describes one
            R,G,B triplet.
        """
        return list(self.rgbs.flatten())

    def to_data(self):
        """ Returns the GBA color representation of this palette.
        
        Returns:
        --------
        data : list
            List of 16-bit values representing the colors of this palette.
        """
        def color_pack(color):
            return (color[0] >> 3) | ((color[1] >> 3) << 5) | ((color[2] << 3) << 10)
        return list(map(color_pack, self.rgbs.astype(int)))


def from_data(data):
    """ Creates a Palette instance based on
    raw binary data.
    
    Parameters:
    -----------
    data : bytes-like or list
        The raw binary data to obtain the rgb values from.
    
    Returns:
    --------
    palette : Palette
        A new Palette instance.
    """
    data = bytes(data)
    values = [struct.unpack_from('<H', data, i)[0] for i in range(0, len(data), 2)]
    values = map(lambda value: ((value & 31) * 8, ((value >> 5) & 31) * 8, ((value >> 10) & 31) * 8), values)
    return Palette(list(values))

def from_file(file_path):
    """ Creates a Palette instance from an image file.
    
    Parameters:
    -----------
    file_path : str
        The file path of the image.

    Returns:
    --------
    palette : Palette
        The palette of the image.
    """
    with open(file_path, 'rb') as f:
        img = PIL.Image.open(f)
        colors = list(bytes(img.palette.palette))
        return Palette(list(zip(colors[0::3], colors[1::3], colors[2::3])))
