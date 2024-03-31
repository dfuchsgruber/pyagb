"""Widget forthe actual map."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsSceneMouseEvent,
    QWidget,
)

import pymap.gui.render as render
from pymap.gui.map_scene import MapScene as BaseMapScene
from pymap.gui.smart_shape import SmartPath

from .level_blocks import level_to_info

if TYPE_CHECKING:
    from .map_widget import MapWidget


class MapScene(BaseMapScene):
    """Scene for the map view."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):  #
        """Initializes the map scene.

        Args:
            map_widget (MapWidget): The map widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(map_widget.main_gui, parent=parent)
        self.map_widget = map_widget
        self.selection_box = None
        # Store the position where a draw happend recently so there are not multiple
        # draw events per block
        self.last_draw = None
        self.smart_drawing = None

    def _update_information_by_position(self, x: int, y: int):
        """Updates the information label based on the position."""
        if not self.map_widget.header_loaded:
            return
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        if (
            border_width <= x < border_width + map_width
            and border_height <= y < border_height + map_height
        ):
            assert self.map_widget.blocks is not None, 'Blocks are not set'
            match self.map_widget.tabs.currentIndex():
                case 0:
                    self.map_widget.info_label.setText(
                        level_to_info(self.map_widget.blocks[y, x, 1])
                    )
                case 1:
                    block = self.map_widget.blocks[y, x]
                    self.map_widget.info_label.setText(
                        (
                            f'x : {hex(x - border_width)}, '
                            f'y : {hex(y - border_height)}, '
                            f'Block : {hex(block[0])}, Level : {hex(block[1])}'
                        )
                    )
                case _:
                    raise ValueError(
                        f'Invalid tab index: {self.map_widget.tabs.currentIndex()}'
                    )
        else:
            self.map_widget.info_label.setText('')

    def _draw_selection_smart(self, x: int, y: int):
        """Draws the selection when the mouse is at position x, y using a smart shape.

        Args:
            x (int): x offset of the mouse, in blocks
            y (int): y offset of the mouse, in blocks
        """
        if not self.map_widget.header_loaded:
            return
        assert self.smart_drawing is not None, 'Smart drawing is not set'
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        if QApplication.keyboardModifiers() != Qt.KeyboardModifier.AltModifier:
            # Smart drawing ends when the ALT key is not pressed anymore
            self.smart_drawing = None
            self.last_draw = None
        elif (
            y - border_height,
            x - border_width,
        ) not in self.smart_drawing:
            for coordinate in self.smart_drawing.complete(y, x):
                self.smart_drawing.append(coordinate)
                # After drawing block with index i we redraw: i - 1, i, i + 1 (=0)
                auto_shape = self.map_widget.auto_shapes_scene.auto_shape[
                    :, :, 0
                ].flatten()
                for idx in (-2, -1, 0):
                    (y, x), shape_idx = self.smart_drawing.get_by_path_idx(idx)
                    self.map_widget.main_gui.set_blocks(
                        x, y, (0,), np.array([[[auto_shape[shape_idx]]]])
                    )

    def _draw_selection(self, x: int, y: int):
        """Draws the selection when the mouse is at position x, y.

        Args:
            x (int): x offset of the mouse, in blocks
            y (int): y offset of the mouse, in blocks
        """
        if not self.map_widget.header_loaded:
            return
        # Check if the selection must be drawn
        self.last_draw = x, y
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        selection = (
            self.map_widget.selection
            if self.map_widget.tabs.currentIndex() == 0
            else self.map_widget.levels_selection
        )
        assert selection is not None, 'Selection is not set'
        if self.smart_drawing is not None:
            self._draw_selection_smart(x, y)
        else:
            self.map_widget.main_gui.set_blocks(
                x - border_width,
                y - border_height,
                self.map_widget.layers,
                selection,
            )

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for hover events on the map image."""
        if not self.map_widget.header_loaded:
            return
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()

        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)

        # Update the information for this position
        if not (
            0 <= x < 2 * border_width + map_width
            and 0 <= y < 2 * border_height + map_height
        ):
            return
        self._update_information_by_position(x, y)

        # Draw the selection if it is at a drawable position
        if (
            x in range(border_width, border_width + map_width)
            and y in range(border_height, border_height + map_height)
            and self.last_draw is not None
            and self.last_draw != (x, y)
        ):
            self._draw_selection(x, y)

        # Update the selection box
        if self.selection_box is not None and self.map_widget.blocks is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.map_widget.set_selection(
                    render.select_blocks(self.map_widget.blocks, *self.selection_box)
                )

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if (
            self.map_widget.main_gui.project is None
            or self.map_widget.main_gui.header is None
        ):
            return

        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()
        pos = event.scenePos()

        x, y = int(pos.x() / 16), int(pos.y() / 16)

        if (
            x < 0
            or x >= 2 * border_width + map_width
            or y < 0
            or y >= 2 * border_height + map_height
        ):
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.selection_box = x, x + 1, y, y + 1
            assert self.map_widget.blocks is not None, 'Blocks are not set'
            self.map_widget.set_selection(self.map_widget.blocks[y : y + 1, x : x + 1])

        elif event.button() == Qt.MouseButton.LeftButton:
            self.last_draw = None
            modifiers = QApplication.keyboardModifiers()

            if modifiers == Qt.KeyboardModifier.ShiftModifier:
                # Replace all blocks of this type with the selection, this is only
                # allowed for 1-block selections
                # Also only one layer is permitted

                layer = self.map_widget.tabs.currentIndex()

                selection = (
                    self.map_widget.selection
                    if self.map_widget.tabs.currentIndex() == 0
                    else self.map_widget.levels_selection
                )
                assert selection is not None, 'Selection is not set'

                selection_height, selection_width, _ = selection.shape

                if (
                    selection_height == 1
                    and selection_width == 1
                    and x in range(border_height, border_width + map_width)
                    and y in range(border_height, border_height + map_height)
                ):
                    assert selection is not None, 'Selection is not set'
                    self.map_widget.main_gui.replace_blocks(
                        x - border_width,
                        y - border_height,
                        layer,
                        selection[0, 0, layer],
                    )

            elif modifiers == Qt.KeyboardModifier.ControlModifier:
                # Flood fill is only allowed for 1-block selections

                # Also only one layer is permitted

                layer = self.map_widget.tabs.currentIndex()

                selection = (
                    self.map_widget.selection
                    if self.map_widget.tabs.currentIndex() == 0
                    else self.map_widget.levels_selection
                )
                assert selection is not None, 'Selection is not set'
                selection_height, selection_width, _ = selection.shape

                if (
                    selection_height == 1
                    and selection_width == 1
                    and x in range(border_height, border_width + map_width)
                    and y in range(border_height, border_height + map_height)
                ):
                    assert selection is not None, 'Selection is not set'
                    self.map_widget.main_gui.flood_fill(
                        x - border_width,
                        y - border_height,
                        layer,
                        selection[0, 0, layer],
                    )

            elif modifiers == Qt.KeyboardModifier.AltModifier:  # Start Smart drawing
                self.smart_drawing = SmartPath()
                self.last_draw = -1, -1  # This triggers the drawing routine
                self.map_widget.undo_stack.beginMacro('Drawing Smart Shapes')
                self.mouseMoveEvent(event)

            else:
                self.last_draw = -1, -1  # This triggers the drawing routine
                self.map_widget.undo_stack.beginMacro('Drawing Blocks')
                self.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if not self.map_widget.header_loaded:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self.selection_box = None
        if event.button() == Qt.MouseButton.LeftButton:
            if self.last_draw is not None:
                self.map_widget.undo_stack.endMacro()
            self.last_draw = None
        if self.smart_drawing:
            # print(f'End smart drawing because of mouse release.')

            self.smart_drawing = None  # Smart drawing ends
