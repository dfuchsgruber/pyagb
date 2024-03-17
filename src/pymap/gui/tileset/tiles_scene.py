"""Scene for the individual tiles."""

from __future__ import annotations

from enum import IntFlag, auto
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)
from typing_extensions import ParamSpec

from .. import render
from .child import TilesetChildWidgetMixin, if_tileset_loaded

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class TileFlip(IntFlag):
    """Flip flags for tiles."""

    HORIZONTAL = auto()
    VERTICAL = auto()


# A static num_pals x num_flips x 64 x 16 array that represents
# the pool of tiles to select from
tiles_pool: npt.NDArray[np.int_] = np.array(
    [
        [
            [
                {
                    'tile_idx': tile_idx,
                    'palette_idx': pal_idx,
                    'horizontal_flip': int((flip & TileFlip.HORIZONTAL) > 0),
                    'vertical_flip': int((flip & TileFlip.VERTICAL) > 0),
                }
                for tile_idx in range(0x400)
            ]
            for flip in range(4)
        ]
        for pal_idx in range(13)
    ],
    dtype=int,
).reshape((13, 4, 64, 16))

for flip in range(4):
    if flip & TileFlip.HORIZONTAL:
        tiles_pool[:, flip, :, :] = tiles_pool[:, flip, :, ::-1]
    if flip & TileFlip.VERTICAL:
        tiles_pool[:, flip, :, :] = tiles_pool[:, flip, ::-1, :]


class TilesScene(QGraphicsScene, TilesetChildWidgetMixin):
    """Scene for the individual tiles."""

    def __init__(self, tileset_widget: TilesetWidget, parent: QWidget | None = None):
        """Initializes the scene.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.selection_box = None
        self.selection_rect = None

    @property
    def is_drawing(self) -> bool:
        """Whether the user is currently drawing."""
        return self.selection_box is not None

    def ensure_selection_rect_visible(self):
        """Ensures that the selection rectangle is visible."""
        if self.selection_rect is not None:
            self.tileset_widget.tiles_scene_view.ensureVisible(
                self.selection_rect.rect()
            )

    def add_selection_rect(self):
        """Adds the selection rectangle."""
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        pen = QPen(color, 1.0 * self.tileset_widget.zoom_slider.value() / 10)
        self.selection_rect = self.addRect(0, 0, 0, 0, pen=pen, brush=QBrush(0))

    @if_tileset_loaded
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event.
        """
        pos = event.scenePos()
        x, y = (
            int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
            int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
        )
        if 16 > x >= 0 and 64 > y >= 0:
            self.tileset_widget.set_info(16 * y + x)
            if self.is_drawing:
                assert self.selection_box is not None
                x0, x1, y0, y1 = self.selection_box
                if x1 != x + 1 or y1 != y + 1:
                    # Redraw the selection
                    self.selection_box = x0, x + 1, y0, y + 1
                    self.select_tiles()
        else:
            self.tileset_widget.set_info(None)

    @if_tileset_loaded
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        pos = event.scenePos()
        x, y = (
            int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
            int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
        )
        if 16 > x >= 0 and 64 > y >= 0:
            if event.button() in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
            ):
                # Select this tile as starting point
                self.selection_box = x, x + 1, y, y + 1
                self.select_tiles()

    @if_tileset_loaded
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton):
            self.selection_box = None

    @if_tileset_loaded
    def select_tiles(self):
        """Updates the selection according to the current selection box."""
        tiles = tiles_pool[
            self.tileset_widget.tiles_palette_combobox.currentIndex(),
            self.tileset_widget.tiles_flip,
        ]
        if self.selection_box is not None:
            self.tileset_widget.set_selection(
                render.select_blocks(tiles, *self.selection_box)
            )
        self.update_selection_rect()

    @if_tileset_loaded
    def update_selection_rect(self):
        """Updates the selection rectangle."""
        if self.selection_rect is None:
            return
        if self.is_drawing:
            assert self.selection_box is not None
            # Redraw the red selection box
            x0, x1, y0, y1 = render.get_box(*self.selection_box)
            scale = 8 * self.tileset_widget.zoom_slider.value() / 10
            self.selection_rect.setRect(
                int(scale * x0),
                int(scale * y0),
                int(scale * (x1 - x0)),
                int(scale * (y1 - y0)),
            )
        else:
            self.selection_rect.setRect(0, 0, 0, 0)
