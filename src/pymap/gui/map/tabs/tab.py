"""Base class for the tabs of the map view."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from ..level import level_to_info

if TYPE_CHECKING:
    from ..map_widget import MapWidget


class MapWidgetTab(QWidget):
    """A tab in the map view."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(parent)
        self.map_widget = map_widget

    @property
    @abstractmethod
    def selected_layers(self) -> NDArray[np.int_]:
        """Get the selected levels."""
        raise NotImplementedError

    @abstractmethod
    def load_project(self) -> None:
        """Load the project."""
        raise NotImplementedError

    @abstractmethod
    def load_header(self) -> None:
        """Load the tab with the information from the map."""
        raise NotImplementedError

    @abstractmethod
    def load_map(self) -> None:
        """Update the map.

        This includes adding the layers to the map scene the tab wants to show.
        """
        raise NotImplementedError

    @abstractmethod
    def set_selection(self, selection: NDArray[np.int_]) -> None:
        """Set the selection."""
        raise NotImplementedError

    @abstractmethod
    def map_scene_mouse_pressed(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for pressing the mouse."""
        raise NotImplementedError

    @abstractmethod
    def map_scene_mouse_moved(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for moving the mouse."""
        raise NotImplementedError

    @abstractmethod
    def map_scene_mouse_released(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for releasing the mouse."""
        raise NotImplementedError

    def get_info_text_by_position(self, x: int, y: int) -> str | None:
        """Get the information text for the position."""
        if not self.map_widget.header_loaded:
            return None
        assert self.map_widget.blocks is not None, 'Blocks are not loaded'
        block, level = self.map_widget.blocks[y, x]
        return f'Block: {hex(block)}, Level: {hex(block)}({level_to_info(level)})'
