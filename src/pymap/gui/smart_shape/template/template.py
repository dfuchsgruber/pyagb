"""Templates for smart shapes."""

import importlib.resources as resources
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QPixmap

from pymap.gui.smart_shape.smart_shape import SmartShape
from pymap.gui.types import RGBAImage


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
    ):
        """Initialize the template.

        Args:
            template_pixmap (str | QPixmap): The picture where blocks are drawn onto
                the template.
            blocks_pixmap (QPixmap, optional): The blocks pixmap. Defaults to None.
            block_tooltips (list[str], optional): The tooltips for each block. Defaults
                to None.
        """
        if isinstance(template_pixmap, str):
            template_pixmap = QPixmap(template_pixmap)
        self.template_pixmap = template_pixmap

        if isinstance(blocks_pixmap, str):
            blocks_pixmap = QPixmap(blocks_pixmap)
        width, height = blocks_pixmap.width(), blocks_pixmap.height()
        assert width % 16 == 0, 'The blocks pixmap width must be a multiple of 8.'
        assert height % 16 == 0, 'The blocks pixmap height must be a multiple of 8.'
        self.block_pixmaps = [
            blocks_pixmap.copy(x * 16, y * 16, 16, 16)
            for y in range(height // 16)
            for x in range(width // 16)
        ]
        if block_tooltips is not None:
            assert len(block_tooltips) == len(self.block_pixmaps), (
                'The number of block tooltips must match the number of blocks.'
            )
        else:
            block_tooltips = [''] * self.num_blocks
        self.block_tooltips = block_tooltips

    @property
    def num_blocks(self) -> int:
        """Return the number of blocks to map this shape to the map."""
        return len(self.block_pixmaps)

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return the dimensions of the template."""
        return self.template_pixmap.width() // 16, self.template_pixmap.height() // 16

    @abstractmethod
    def generate_blocks(
        self,
        smart_shape: SmartShape,
        map_blocks: RGBAImage,
    ) -> tuple[RGBAImage, tuple[NDArray[np.int_], ...]]:
        """Generates map blocks for the given smart shape.

        Args:
            smart_shape (SmartShape): The smart shape to generate blocks for.
            map_blocks (RGBAImage): The blocks of the map.

        Returns:
            RGBAImage: The updated buffer.
            RGBAImage: Which blocks in the buffer a to be changed.
        """
        raise NotImplementedError
