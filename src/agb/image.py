"""Image module to provide a png-based image wrapper."""

from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImagePalette
import png  # type: ignore
from colormath.color_conversions import convert_color  # noqa: E402 # type: ignore
from colormath.color_diff import delta_e_cie2000  # noqa: E402 # type: ignore
from colormath.color_objects import LabColor, sRGBColor  # noqa: E402 # type: ignore

from .palette import Palette

# Monkey patching for colormath
np.asscalar = float  # type: ignore


class Image:
    """Class that represents a png-based image."""

    SUPPORTED_DEPTHS = {1, 2, 4, 8}

    def __init__(
        self, data: Sequence[int] | None, width: int, height: int, depth: int = 4
    ):
        """Initializes an image from binary data.

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
        assert depth in self.SUPPORTED_DEPTHS, (
            f'Invalid depth {depth}, must be in {self.SUPPORTED_DEPTHS}'
        )
        assert width % 8 == 0, 'Width is not a multiple of 8'
        assert height % 8 == 0, 'Height is not a multiple of 8'
        self.depth = depth
        self.width = width
        self.height = height
        self.data = np.zeros((width, height), dtype=np.int_)
        if data is not None:
            for x, y in product(range(width), range(height)):
                byte_index, bit_index = self._position_to_data_idx(x, y)
                self.data[x, y] = (data[byte_index] >> bit_index) & (2**depth - 1)

    def to_binary(self) -> bytearray:
        """Returns the raw binary data of the image in GBA tile format.

        Note that this has linear complexity in the number of pixels.

        Returns:
        --------
        binary : bytearray
            The binary representation of the picture in GBA tile format.
        """
        # Pack the data to a dense binary format
        binary = bytearray([0] * (self.width * self.height * self.depth // 8))
        for x, y in product(range(self.width), range(self.height)):
            byte_index, bit_index = self._position_to_data_idx(x, y)
            binary[byte_index] |= (self.data[x, y] & (2**self.depth - 1)) << bit_index
        return binary

    def _position_to_data_idx(self, x: int, y: int) -> tuple[int, int]:
        """Converts a position to the data index and bit index.

        Parameters:
        -----------
        x : int
            The x position.
        y : int
            The y position.

        Returns:
        --------
        data_idx : int
            Which index in the byte array the position is located.
        bit_idx : int
            At which bit in the byte the position is located.
        """
        tile_x, tile_y = x >> 3, y >> 3
        tile_idx = tile_y * (self.width >> 3) + tile_x
        tile_off_x, tile_off_y = x & 7, y & 7
        data_idx = (
            tile_idx * 8 * self.depth
            + tile_off_y * self.depth
            + tile_off_x * self.depth // 8
        )
        return data_idx, 8 - self.depth - tile_off_x * self.depth % 8

    def apply_palette(self, palette_src: Palette, palette_target: Palette):
        """Applies a palette to the image.

        It remaps the colors of the image to the colors of the palette that
        are perceptually closest to the original colors.

        Parameters:
        -----------
        palette_src: agb.palette.Palette
            The source palette.
        palette_target: agb.palette.Palette
            The target palette.
        """
        assert len(palette_src) <= 2**self.depth, (
            'Source palette is too large for image depth!'
        )
        assert len(palette_target) <= 2**self.depth, (
            'Target palette is too large for image depth!'
        )

        # remap palette
        palette_map = np.zeros(2**self.depth, dtype=int)

        for color_idx in range(1, 16):
            best_distance: float = float(np.inf)
            rgb_src = sRGBColor(*(palette_src.rgbs[color_idx] / 256))
            lab_src: float = convert_color(rgb_src, LabColor)  # type: ignore
            for target_idx in range(1, len(palette_target)):
                rgb_target = sRGBColor(*(palette_target.rgbs[target_idx] / 256))
                lab_target: float = convert_color(rgb_target, LabColor)  # type: ignore
                distance: float = delta_e_cie2000(lab_src, lab_target)  # type: ignore
                if distance < best_distance:
                    best_distance = distance  # type: ignore
                    palette_map[color_idx] = target_idx
        self.data = palette_map[self.data]

    def to_pil_image(
        self, palette: Sequence[int], transparent: int | None = 0
    ) -> PIL.Image.Image:
        """Creates a Pilow image based on the image resource.

        Parameters:
        -----------
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        transparent : int or None
            The color index of the palette which resembles transparency, i.e. alpha
            value of zero. If None, no alpha is applied to the image.

        Returns:
        --------
        image : PIL.Image.Image
            The Pilow image.
        """
        img = PIL.Image.new('P', (self.width, self.height))
        img.putdata(self.data.T.flatten().tolist())  # type: ignore
        img.putpalette(palette)  # type: ignore
        img = img.convert('RGBA')
        if transparent is not None:
            alpha_color = list(palette[transparent * 3 : (transparent + 1) * 3])
            img.putdata(  # type: ignore
                [
                    (c[0], c[1], c[2], 0) if list(c)[:3] == alpha_color[:3] else c  # type: ignore
                    for c in img.getdata()  # type: ignore
                ]
            )
        return img

    def to_rgba(
        self, palette: Sequence[int], transparent: int | None = 0
    ) -> npt.NDArray[np.int_]:
        """Creates a RGBA image based on the image resource.

        Parameters:
        -----------
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        transparent : int or None
            The color index of the palette which resembles transparency, i.e. alpha
            value of zero. If None, no alpha is applied to the image.

        Returns:
        --------
        image : np.ndarray, shape [h, w, 4]
            The RGBA image.
        """
        palette_ndarray = np.array(palette, dtype=np.int_).reshape(-1, 3)
        palette_rgba = np.zeros((len(palette_ndarray), 4), dtype=np.int_)
        palette_rgba[:, :3] = palette_ndarray
        palette_rgba[:, 3] = 255
        if transparent is not None:
            palette_rgba[transparent, 3] = 0
        return palette_rgba[self.data.T]

    def save(
        self, path: str, palette: bytes | Sequence[int] | PIL.ImagePalette.ImagePalette
    ):
        """Saves this image with a given palette.

        Parameters:
        -----------
        path : str
            The path to save the image at.
        palette : list or byte-like
            Sequence where each groups of 3 represent the R,G,B values
        """
        img = PIL.Image.new('P', (self.width, self.height))
        img.putdata(self.data.T.flatten().tolist())  # type: ignore
        img.putpalette(palette)  # type: ignore
        img.save(str(Path(path)))  # type: ignore


def from_file(file_path: str) -> tuple[Image, Palette]:
    """Creates an Image instance from a png file.

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
        width, height, data, attributes = reader.read()  # type: ignore
        attributes: dict[str, int | str] = attributes
        bitdepth = attributes['bitdepth']
        assert isinstance(bitdepth, int), f'Bitdepth {bitdepth} is not an integer'
        assert bitdepth & (bitdepth - 1) == 0, (
            f'Bitdepth {bitdepth} is not a power of 2'
        )
        image = Image(None, width, height, depth=bitdepth)
        image.data = np.array([*data]).T
        colors = attributes['palette']
        _palette = Palette(colors, size=len(colors))  # type: ignore
        return image, _palette
