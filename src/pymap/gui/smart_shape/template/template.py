"""Templates for smart shapes."""

from __future__ import annotations

import importlib.resources as resources
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QPixmap

from pymap.gui.smart_shape.smart_shape import SmartShape
from pymap.gui.types import Tilemap

if TYPE_CHECKING:
    from pymap.gui.rgba_image import QRGBAImage


class SmartShapeTemplate:
    """Base class for all potential smart shape templates.

    Args:
        template_pixmap (str | QPixmap): The picture where blocks are drawn onto
            the template.
        blocks_pixmap (str | QPixmap, optional): The meta-blocks that can be drawn onto
            the map.
        block_tooltips (list[str], optional): The tooltips for each meta-block. Defaults
            to None.
    """

    def __init__(
        self,
        template_pixmap: str | QPixmap,
        blocks_pixmap: str | QPixmap = str(
            resources.files('pymap.gui.smart_shape.template').joinpath(
                'default_template_blocks.png'
            )
        ),
        block_tooltips: list[str] | None = ['None', 'Auto-Fill'],
        blocks_opacity: float = 0.5,
    ):
        """Initialize the template.

        Args:
            template_pixmap (str | QPixmap): The picture where blocks are drawn onto
                the template.
            blocks_pixmap (QPixmap, optional): The blocks pixmap. Defaults to None.
            block_tooltips (list[str], optional): The tooltips for each block. Defaults
                to None.
            blocks_opacity (float, optional): The opacity of the blocks.
                Defaults to 0.5.
        """
        from pymap.gui.render import QImage_to_ndarray, split_image_into_tiles

        if isinstance(template_pixmap, str):
            self._template_pixmap = QPixmap(template_pixmap)
        else:
            self._template_pixmap = template_pixmap

        if isinstance(blocks_pixmap, str):
            blocks_pixmap = QPixmap(blocks_pixmap)
        self.block_images = split_image_into_tiles(
            QImage_to_ndarray(
                blocks_pixmap.toImage(),
            ),
            tile_size=16,
        ).reshape((-1, 16, 16, 4))

        width, height = blocks_pixmap.width(), blocks_pixmap.height()
        assert width % 16 == 0, 'The blocks pixmap width must be a multiple of 8.'
        assert height % 16 == 0, 'The blocks pixmap height must be a multiple of 8.'

        if block_tooltips is not None:
            assert len(block_tooltips) == self.num_blocks, (
                'The number of block tooltips must match the number of blocks.'
            )
        else:
            block_tooltips = [''] * self.num_blocks
        self.block_tooltips = block_tooltips
        self.blocks_opacity = blocks_opacity

    @property
    def template_qrgba_image(self) -> QRGBAImage:
        """Return the template as a QRGBAImage."""
        from pymap.gui.render import QImage_to_ndarray
        from pymap.gui.rgba_image import QRGBAImage

        template_rgba = QImage_to_ndarray(
            self._template_pixmap.toImage(),
        )
        return QRGBAImage(
            template_rgba,
        )

    @property
    def num_blocks(self) -> int:
        """Return the number of blocks to map this shape to the map."""
        return self.block_images.shape[0]

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the dimensions of the template."""
        return (
            self._template_pixmap.width() // 16,
            self._template_pixmap.height() // 16,
        )

    @abstractmethod
    def generate_blocks(
        self,
        smart_shape: SmartShape,
        map_blocks: Tilemap,
    ) -> tuple[Tilemap, tuple[NDArray[np.int_], ...]]:
        """Generates map blocks for the given smart shape.

        Args:
            smart_shape (SmartShape): The smart shape to generate blocks for.
            map_blocks (RGBAImage): The blocks of the map.

        Returns:
            RGBAImage: The updated buffer.
            RGBAImage: Which blocks in the buffer are to be changed.
        """
        raise NotImplementedError
