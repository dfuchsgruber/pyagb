""" Module to provide a png-based image wrapper."""

import PIL.Image
import numpy as np
from itertools import product
import png
from . import palette

class Image:
    """ Class that represents a png-based image."""

    def __init__(self, data, width, height, depth=4):
        """ Initializes an image from binary data.
        
        Parameters:
        -----------
        data : bytes-like or None
            Binary raw pixel data in GBA format.
            If None the image will be empty.
        width : int
            The width of the picture.
        height : int
            The height of the picture.
        bpp : 4 or 8
            Bits per pixel (depth). 
        """
        self.data = np.zeros((width, height), dtype=np.int)
        if data is not None:
            # Unpack the data
            tile_width = width // 8
            for idx in range(len(data)):
                if depth == 4:
                    tile_idx = idx // 32 # (8 * 8 / 2) = 32 bytes per tile
                    tile_pos = idx % 32
                    x = 8 * (tile_idx % tile_width)
                    y = 8 * (tile_idx // tile_width)
                    y += tile_pos // 4
                    x += 2 * (tile_pos % 4)
                    # print(f'Tile {idx} -> {x}, {y}')
                    self.data[x, y] = data[idx] & 0xF
                    self.data[x + 1, y] = data[idx] >> 4
                elif depth == 8:
                    tile_idx = idx // 64 # (8 * 8) = 64 bytes per tile
                    tile_pos = idx % 64
                    x = 8 * (tile_idx % tile_width)
                    y = 8 * (tile_idx // tile_width)
                    y += tile_pos // 8
                    x += tile_pos % 8
                    self.data[x, y] = data[idx]
                else:
                    raise RuntimeError('Invalid image depth. Only depths 4 and 8 are supported!')

        self.depth = depth
        self.width = width
        self.height = height
    
    def to_binary(self):
        """ Returns the raw binary data of the image in GBA tile format.
        Note that this has linear complexity in the number of pixels.

        Returns:
        --------
        binary : bytearray
            The binary representation of the picture in GBA tile format.
        """
        # Pack the data to a dense binary format
        binary = bytearray([0] * (self.width * self.height * self.depth // 8))
        for x, y in product(range(self.width), range(self.height)):
            tile_x, tile_y = x >> 3, y >> 3
            tile_idx = tile_y * (self.width >> 3) + tile_x
            tile_off_x, tile_off_y = x & 7, y & 7
            if self.depth == 4:
                idx = tile_idx << 5 # (8 * 8 / 2) = 32 bytes per tile
                idx += ((tile_off_y << 3) + tile_off_x) >> 1
                # print(f'{x}, {y} -> {idx} [upper={tile_off_x & 1 > 0}]')
                if tile_off_x & 1 > 0:
                    binary[idx] |= self.data[x, y] << 4
                else:
                    binary[idx] |= self.data[x, y]
            elif self.depth == 8:
                idx = tile_idx << 6 # (8 * 8) = 64 bytes per tile
                idx += tile_off_y << 3 + tile_off_x
                binary[idx] = self.data[x, y]
            else:
                raise RuntimeError('Invalid image depth. Only depths 4 and 8 are supported!')
        return binary
        
    def save(self, file_path, palette):
        """ Saves the image as png-file.
        
        Parameters:
        -----------
        file_path : string
            The file path to save the image at.
        palette : agb.palette.Palette
            The palette to use.
        """
        with open(file_path, 'wb') as f:
            writer = png.Writer(self.width, self.height, palette=palette.rgbs, bitdepth=self.depth)
            writer.write(f, self.data.T)



    def to_pil_image(self, palette):
        """ Creates a Pilow image based on the image resource.
        
        Parameters:
        -----------
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        
        Returns:
        --------
        image : PIL.Image
            The Pilow image.
        """
        img = PIL.Image.new('P', (self.width, self.height))
        img.putdata(self.data.T.flatten())
        img.putpalette(palette)
        return img


def from_file(file_path):
    """ Creates an Image instance from a png file.
    
    Parameters:
    -----------
    file_path : str
        The path to the png file.
    
    Returns:
    --------
    image : Image
        The image instance
    palette : agb.palette
        The palette of the image
    """
    with open(file_path, 'rb') as f:
        reader = png.Reader(f)
        width, height, data, attributes = reader.read()
        image = Image(None, width, height, depth=attributes['bitdepth'])
        image.data = np.array([*data]).T
        colors = attributes['palette']
        _palette = palette.Palette(list(zip(colors[0::3], colors[1::3], colors[2::3])))
        return image, _palette