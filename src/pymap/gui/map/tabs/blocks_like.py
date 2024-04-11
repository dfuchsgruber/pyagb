"""Base class for blocks-like tabs that support selection, flood-filling, etc."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pymap.gui import render
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsSceneMouseEvent,
    QWidget,
)

from .tab import MapWidgetTab

if TYPE_CHECKING:
    from ..map_widget import MapWidget


class CursorState(Enum):
    """State of the cursor."""

    DEFAULT = 0
    DRAW = 1
    SELECT = 2


class BlocksLikeTab(MapWidgetTab):
    """Tabs with block like functionality.

    They have a selection of blocks that can be drawn to the map. They also
    support flood filling and replacement. The selection can be taken from the map as
    well or set natively via the `set_selection` method to `selection`.
    """

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)
        self._position_last_drawn = None
        self._map_selected_rectangle = None
        self.selection: NDArray[np.int_] | None = None
        self._cursor_state = CursorState.DEFAULT

    @property
    def connectivity_layer(self) -> int:
        """The layer for connectivity.

        Use for flood filling, replacement, etc.
        """
        raise NotImplementedError

    def set_selection(self, selection: NDArray[np.int_]) -> None:
        """Sets the selection.

        Args:
            selection (NDArray[np.int_]): The selection.
        """
        raise NotImplementedError

    def load_project(self):
        """Loads the project."""
        self.set_selection(np.zeros((1, 1, 2), dtype=int))

    def load_header(self):
        """Loads the tab."""
        if self.map_widget.header_loaded:
            assert self.selection is not None
            self.set_selection(self.selection)

    def map_scene_mouse_pressed(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for pressing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if event.button() == Qt.MouseButton.RightButton:
            assert self.map_widget.blocks is not None, 'Blocks are not loaded'
            self._cursor_state = CursorState.SELECT
            self._map_selected_rectangle = (x, x + 1, y, y + 1)
            self.set_selection(self.map_widget.blocks[y : y + 1, x : x + 1])
        elif event.button() == Qt.MouseButton.LeftButton:
            self._position_last_drawn = None
            self._cursor_state = CursorState.DRAW
            match QApplication.keyboardModifiers():
                case Qt.KeyboardModifier.ShiftModifier:
                    # Replace all blocks of this type with the selection
                    self.map_scene_mouse_pressed_shift(x, y)
                case Qt.KeyboardModifier.ControlModifier:
                    # Flood fill with the selection
                    self.map_scene_mouse_pressed_control(x, y)
                case _:
                    # Pretend that the previous draw was outside the drawable region
                    # to trigger a draw for the initial position as well
                    self._position_last_drawn = -1, -1
                    self.map_widget.undo_stack.beginMacro('Draw Blocks')
                    self.map_scene_mouse_moved(event, x, y)

    def map_scene_mouse_pressed_shift(self, x: int, y: int):
        """Event handler for pressing the mouse with the shift key pressed.

        This is replace the current block with the selection.

        Args:
            x (int): The x coordinate of the mouse in map coordinates (with border padding).
            y (int): The y coordinate of the mouse in map coordinates (with border padding).
        """
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        assert self.selection is not None, 'No selection'
        if (
            self.selection.shape[0] == 1
            and self.selection.shape[1] == 1
            and x in range(border_width, map_width + border_width)
            and y in range(border_height, map_height + border_height)
        ):
            self.map_widget.main_gui.replace_blocks(
                x - border_width,
                y - border_height,
                self.connectivity_layer,
                self.selection[0, 0, self.connectivity_layer],
            )

    def map_scene_mouse_pressed_control(self, x: int, y: int):
        """Event handler for pressing the mouse with the control key pressed.

        Args:
            x (int): The x coordinate of the mouse in map coordinates (with border padding).
            y (int): The y coordinate of the mouse in map coordinates (with border padding).
        """
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        assert self.selection is not None, 'No selection'
        if (
            self.selection.shape[0] == 1
            and self.selection.shape[1] == 1
            and x in range(border_width, map_width + border_width)
            and y in range(border_height, map_height + border_height)
        ):
            self.map_widget.main_gui.flood_fill(
                x - border_width,
                y - border_height,
                self.connectivity_layer,
                self.selection[0, 0, self.connectivity_layer],
            )

    def map_scene_mouse_moved(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for moving the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        match self._cursor_state:
            case CursorState.SELECT:
                assert self._map_selected_rectangle is not None, 'No selection box'
                assert self.map_widget.blocks is not None, 'Blocks are not loaded'
                x0, x1, y0, y1 = self._map_selected_rectangle
                if x1 != x + 1 or y1 != y + 1:
                    self._map_selected_rectangle = x0, x + 1, y0, y + 1
                    self.set_selection(
                        render.select_blocks(
                            self.map_widget.blocks, *self._map_selected_rectangle
                        )
                    )
            case CursorState.DRAW:
                if self._position_last_drawn == (x, y):
                    return  # No need to draw the same position again
                map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
                (
                    border_width,
                    border_height,
                ) = self.map_widget.main_gui.get_border_padding()
                if x in range(border_width, map_width + border_width) and y in range(
                    border_height, map_height + border_height
                ):
                    assert self.selection is not None, 'No selection'
                    self._position_last_drawn = (x, y)
                    self.map_widget.main_gui.set_blocks_at(
                        x - border_width,
                        y - border_height,
                        self.selected_layers,
                        self.selection,
                    )
            case _:
                ...

    def map_scene_mouse_released(
        self, event: QGraphicsSceneMouseEvent, x: int, y: int
    ) -> None:
        """Event handler for releasing the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The event.
            x (int): x coordinate of the mouse in map coordinates (with border padding)
            y (int): y coordinate of the mouse in map coordinates (with border padding)
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if self._position_last_drawn is not None:
                self.map_widget.undo_stack.endMacro()
            self._position_last_drawn = None
            self._cursor_state = CursorState.DEFAULT
        elif event.button() == Qt.MouseButton.RightButton:
            self._map_selected_rectangle = None
            self._cursor_state = CursorState.DEFAULT
