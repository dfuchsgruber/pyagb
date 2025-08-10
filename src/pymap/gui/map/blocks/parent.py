"""Widget for the blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pymap.gui.types import Tilemap

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui

    from .scene import BlocksScene


class BlocksSceneParent(Protocol):
    """Parent of a blocks scene, in which the blocks of this scene are used."""

    def set_selection(self, selection: Tilemap) -> None:
        """Sets the selection according to what is selected in this scene.

        Args:
            selection (Tilemap): The selection array.
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
