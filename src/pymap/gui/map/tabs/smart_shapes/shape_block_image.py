"""Generates the smart shape block map."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
)

from pymap.gui.blocks import block_idxs_to_pixmaps
from pymap.gui.render import BlockImages
from pymap.gui.smart_shape.smart_shape import SmartShape


def smart_shape_get_block_image(
    smart_shape: SmartShape,
    block_images: BlockImages,
    opacity: float = 0.5,
) -> NDArray[Any]:
    """Gets the block images for a smart shape."""
    pixmaps = block_idxs_to_pixmaps(smart_shape.blocks, block_images)
    for _, pixmap_item in np.ndenumerate(pixmaps):
        assert isinstance(pixmap_item, QGraphicsPixmapItem)
        pixmap_item.setOpacity(opacity)
    return pixmaps
