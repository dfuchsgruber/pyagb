""" Module to provide a png-based image wrapper."""

import PIL.Image
import numpy as np
from itertools import product
import png
from . import palette
from pathlib import Path

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

np.asscalar = float # patches deprecated function used by colormath
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
        self.data = np.zeros((width, height), dtype=int)
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
                    if x < width and y < height:
                        self.data[x, y] = data[idx] & 0xF
                        self.data[x + 1, y] = data[idx] >> 4
                elif depth == 8:
                    tile_idx = idx // 64 # (8 * 8) = 64 bytes per tile
                    tile_pos = idx % 64
                    x = 8 * (tile_idx % tile_width)
                    y = 8 * (tile_idx // tile_width)
                    y += tile_pos // 8
                    x += tile_pos % 8
                    if x < width and y < height:
                        self.data[x, y] = data[idx]
                else:
                    raise RuntimeError('Invalid image depth. Only depths 4 and 8 are supported!')

        self.depth = depth
        self.width = width
        self.height = height
    
    def apply_palette(self, palette_src: palette.Palette, palette_target: palette.Palette):
        """ Applies a palette to the image. That is, it remaps the colors of the image to the colors of the palette that
        are perceptually closest to the original colors.
        
        Parameters:
        -----------
        palette_src: agb.palette.Palette
            The source palette.
        palette_target: agb.palette.Palette
            The target palette.
        """
        assert len(palette_src) <= 2**self.depth, 'Source palette is too large for image depth!'
        assert len(palette_target) <= 2**self.depth, 'Target palette is too large for image depth!'
        
        # remap palette
        palette_map = np.zeros(2**self.depth, dtype=int)
        
        for color_idx in range(1, 16):
            best_distance = np.inf
            rgb_src = sRGBColor(*(palette_src.rgbs[color_idx] / 256))
            lab_src = convert_color(rgb_src, LabColor)
            for target_idx in range(1, len(palette_target)):
                
                rgb_target = sRGBColor(*(palette_target.rgbs[target_idx] / 256))
                lab_target = convert_color(rgb_target, LabColor)
                distance = delta_e_cie2000(lab_src, lab_target)
                if distance < best_distance:
                    best_distance = distance
                    palette_map[color_idx] = target_idx
        self.data = palette_map[self.data]
    
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

    def to_pil_image(self, palette, transparent=0):
        """ Creates a Pilow image based on the image resource.
        
        Parameters:
        -----------
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        transparent : int or None
            The color index of the palette which resembles transparency, i.e. alpha value of zero.
            If None, no alpha is applied to the image.
        
        Returns:
        --------
        image : PIL.Image
            The Pilow image.
        """
        img = PIL.Image.new('P', (self.width, self.height))
        img.putdata(self.data.T.flatten())
        img.putpalette(palette)
        img = img.convert('RGBA')
        if transparent is not None:
            alpha_color = palette[transparent * 3 : (transparent + 1) * 3]
            img.putdata([
                (c[0], c[1], c[2], 0) if c[0] == alpha_color[0] and c[1] == alpha_color[1] and c[2] == alpha_color[2] else c
                for c in img.getdata()
            ])
        return img

    def save(self, path, palette):
        """ Saves this image with a given palette. 
        
        Parameters:
        -----------
        path : str
            The path to save the image at.
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        """
        img = PIL.Image.new('P', (self.width, self.height))
        img.putdata(self.data.T.flatten())
        img.putpalette(palette)
        img.save(str(Path(path)))

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
    with open(Path(file_path), 'rb') as f:
        reader = png.Reader(f)
        width, height, data, attributes = reader.read()
        image = Image(None, width, height, depth=attributes['bitdepth'])
        image.data = np.array([*data]).T
        colors = attributes['palette']
        _palette = palette.Palette(colors)
        return image, _palette