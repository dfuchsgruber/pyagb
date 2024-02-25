"""Scene for the selected tile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from .tileset import TilesetWidget

from .child import TilesetChildWidgetMixin, if_tileset_loaded


class SelectionScene(QtWidgets.QGraphicsScene, TilesetChildWidgetMixin):
    """Scene for the selected tiles."""

    def __init__(
        self, tileset_widget: TilesetWidget, parent: QtWidgets.QWidget | None = None
    ):
        """Initializes the scene.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QtWidgets.QWidget | None, optional): The parent widget.
                Defaults to None.
        """
        super().__init__(parent=parent)
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.tileset_widget = tileset_widget

    @if_tileset_loaded
    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.tileset_widget.tileset_loaded:
            return
        pos = event.scenePos()
        x, y = (
            int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
            int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
        )
        height, width = self.tileset_widget.selection.shape
        if width > x >= 0 and height > y >= 0:
            self.tileset_widget.set_info(
                self.tileset_widget.selection[y, x]['tile_idx']
            )
        else:
            self.tileset_widget.set_info(None)
