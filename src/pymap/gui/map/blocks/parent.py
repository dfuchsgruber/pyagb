"""Widget for the blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from agb.model.type import IntArray
from PIL.ImageQt import ImageQt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
)

import pymap.gui.render as render

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

    from .scene import BlocksScene


class BlocksSceneParent(Protocol):
    """Parent of a blocks scene, in which the blocks of this scene are used."""

    def set_selection(self, selection: IntArray) -> None:
        """Sets the selection according to what is selected in this scene.

        Args:
            selection (IntArray): The selection array.
        """
        ...

    def set_info_text(self, text: str) -> None:
        """Sets the text of the info label.

        Args:
            text (str): The text to set.
        """
        ...

    @property
    def main_gui(self) -> PymapGui:
        """Returns the main GUI.

        Returns:
            PymapGui: The main GUI.
        """
        ...


class BlocksSceneParentMixin:
    """Mixin for a blocks scene parent.

    It extends with functionality to load blocks from the main gui.
    """

    @property
    def blocks_scene(self) -> BlocksScene:
        """The blocks scene.

        Returns:
            BlocksScene: The blocks scene.
        """
        return self._blocks_scene

    @blocks_scene.setter
    def blocks_scene(self, value: BlocksScene) -> None:
        """Sets the blocks scene.

        Args:
            value (BlocksScene): The blocks scene.
        """
        self._blocks_scene = value

    def load_blocks(self):
        """Loads the block pool."""
        self.blocks_scene.clear()
        main_gui = self.blocks_scene.blocks_scene_parent.main_gui
        if not main_gui.footer_loaded:
            return
        map_blocks = main_gui.block_images
        assert map_blocks is not None, 'Blocks are not loaded'
        self.blocks_image = QPixmap.fromImage(ImageQt(render.draw_blocks(map_blocks)))
        item = QGraphicsPixmapItem(self.blocks_image)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        # This triggers segfaults, because the pixmap item is deleted before the lambda is called
        # item.hoverLeaveEvent = (
        #     lambda event: self.blocks_scene.blocks_scene_parent.set_info_text('')
        # )
