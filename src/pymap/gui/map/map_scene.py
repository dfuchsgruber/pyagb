"""Widget forthe actual map."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QGraphicsSceneMouseEvent,
    QWidget,
)

from pymap.gui.map_scene import MapScene as BaseMapScene

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

    def event_coordinates_to_padded_map_coordinates(
        self,
        event: QGraphicsSceneMouseEvent,
    ) -> tuple[int, int] | None:
        """Converts the event coordinates to the padded map coordinates.

        Args:
            event (QGraphicsSceneMouseEvent): The event.

        Returns:
            tuple[int, int] | None: The padded map coordinates or None if the event is
            outside the map.
        """
        if not self.map_widget.header_loaded:
            return None
        map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
        border_width, border_height = self.map_widget.main_gui.get_border_padding()

        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)

        # Update the information for this position
        if not (
            0 <= x < 2 * border_width + map_width
            and 0 <= y < 2 * border_height + map_height
        ):
            return None
        else:  # Return the padded map coordinates
            return x, y

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        if not self.map_widget.header_loaded:
            return
        map_coordinates = self.event_coordinates_to_padded_map_coordinates(event)
        if map_coordinates is None:
            return
        self.map_widget.tabs.currentWidget().map_scene_mouse_pressed(
            event, *map_coordinates
        )

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if not self.map_widget.header_loaded:
            return
        map_coordinates = self.event_coordinates_to_padded_map_coordinates(event)
        if map_coordinates is None:
            return
        self.map_widget.tabs.currentWidget().map_scene_mouse_released(
            event, *map_coordinates
        )

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        if not self.map_widget.header_loaded:
            return
        map_coordinates = self.event_coordinates_to_padded_map_coordinates(event)
        if map_coordinates is None:
            info_text = ''
        else:
            info_text = self.map_widget.tabs.currentWidget().get_info_text_by_position(
                *map_coordinates
            )
        if info_text is not None:
            self.map_widget.info_label.setText(info_text)
        if map_coordinates is not None:
            self.map_widget.tabs.currentWidget().map_scene_mouse_moved(
                event, *map_coordinates
            )
