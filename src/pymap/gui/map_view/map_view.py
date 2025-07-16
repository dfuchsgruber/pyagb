"""MapView class for displaying a tilemap with various layers and functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
)

from agb.model.type import ModelValue
from pymap.gui.blocks import (
    compute_blocks,
    insert_connection,
)

from .blocks import MapViewLayerBlocks
from .connections import MapViewLayerConnections
from .events import MapViewLayerEvents
from .grid import MapViewLayerGrid
from .levels import MapViewLayerLevels
from .selected_event import MapViewLayerSelectedEvent
from .smart_shapes import MapViewLayerSmartShapes

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


from .layer import MapViewLayer, VisibleLayer


class MapView(QGraphicsView):
    """A QGraphicsView that displays a tilemap with a transparent background."""

    def __init__(self, main_gui: PymapGui):
        """Initializes the map view."""
        super().__init__()
        self.main_gui = main_gui
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setScene(QGraphicsScene(self))
        self.layers: dict[VisibleLayer, MapViewLayer] = {
            VisibleLayer.BLOCKS: MapViewLayerBlocks(self),
            VisibleLayer.LEVELS: MapViewLayerLevels(self),
            VisibleLayer.CONNECTION_RECTANGLES: MapViewLayerConnections(self),
            VisibleLayer.EVENTS: MapViewLayerEvents(self),
            VisibleLayer.SELECTED_EVENT: MapViewLayerSelectedEvent(self),
            VisibleLayer.GRID: MapViewLayerGrid(self),
            VisibleLayer.SMART_SHAPE: MapViewLayerSmartShapes(self),
        }

    @property
    def blocks(self) -> MapViewLayerBlocks:
        """Returns the blocks layer."""
        return cast(MapViewLayerBlocks, self.layers[VisibleLayer.BLOCKS])

    @property
    def levels(self) -> MapViewLayerLevels:
        """Returns the levels layer."""
        return cast(MapViewLayerLevels, self.layers[VisibleLayer.LEVELS])

    @property
    def connections(self) -> MapViewLayerConnections:
        """Returns the connections layer."""
        return cast(
            MapViewLayerConnections, self.layers[VisibleLayer.CONNECTION_RECTANGLES]
        )

    @property
    def events(self) -> MapViewLayerEvents:
        """Returns the events layer."""
        return cast(MapViewLayerEvents, self.layers[VisibleLayer.EVENTS])

    @property
    def selected_event(self) -> MapViewLayerSelectedEvent:
        """Returns the selected event layer."""
        return cast(MapViewLayerSelectedEvent, self.layers[VisibleLayer.SELECTED_EVENT])

    @property
    def smart_shapes(self) -> MapViewLayerSmartShapes:
        """Returns the smart shapes layer."""
        return cast(MapViewLayerSmartShapes, self.layers[VisibleLayer.SMART_SHAPE])

    @property
    def grid(self) -> MapViewLayerGrid:
        """Returns the grid layer."""
        return cast(MapViewLayerGrid, self.layers[VisibleLayer.GRID])

    def compute_visible_blocks(self):
        """Computes the visible block tilemap."""
        assert self.main_gui.project is not None
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()

        # Crop the visible blocks from all blocks including the border
        self.visible_blocks = compute_blocks(
            self.main_gui.footer, self.main_gui.project
        )  #
        connections = self.main_gui.get_connections()
        for connection in connections:
            insert_connection(
                self.visible_blocks,
                connection,
                self.main_gui.footer,
                self.main_gui.project,
            )
        visible_width, visible_height = (
            map_width + 2 * padded_width,
            map_height + 2 * padded_height,
        )
        invisible_border_width, invisible_border_height = (
            (self.visible_blocks.shape[1] - visible_width) // 2,
            (self.visible_blocks.shape[0] - visible_height) // 2,
        )
        self.visible_blocks = self.visible_blocks[
            invisible_border_height : self.visible_blocks.shape[0]
            - invisible_border_height,
            invisible_border_width : self.visible_blocks.shape[1]
            - invisible_border_width,
        ]

    def update_grid(self):
        """Updates the grid of the scene."""
        ...

    def load_project(self):
        """Loads the project."""
        ...

    def load_map(self):
        """(Re)loads the map into the view."""
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.compute_visible_blocks()
        for layer in self.layers.values():
            layer.load_map()
            if layer.item is not None:
                self.scene().addItem(layer.item)

        # Set the size of the scene
        padded_width, padded_height = self.main_gui.get_border_padding()
        map_width, map_height = self.main_gui.get_map_dimensions()
        self.scene().setSceneRect(
            0,
            0,
            16 * (2 * padded_width + map_width),
            16 * (2 * padded_height + map_height),
        )

    def update_visible_layers(self, visible_layers: VisibleLayer):
        """Shows / Hides certain layers.

        Args:
            visible_layers (Layer): Mask for visible layers.
        """
        self.visible_layers = visible_layers
        for layer_name, layer in self.layers.items():
            if layer.item is not None:
                layer.item.setVisible((visible_layers & layer_name) > 0)

    @staticmethod
    def pad_coordinates(
        x: ModelValue, y: ModelValue, padded_x: int, padded_y: int
    ) -> tuple[int, int]:
        """Tries to transform the text string of an event to integer coordinates."""
        x, y = str(x), str(y)  # This enables arbitrary bases
        try:
            x = int(x, 0)
            y = int(y, 0)
        except ValueError:
            return (
                -10000,
                -10000,
            )  # This is hacky but prevents the events from being rendered
        return (x + padded_x), (y + padded_y)

    @staticmethod
    def fix_rectangle(
        x: int, y: int, width: int, height: int, max_width: int, max_height: int
    ) -> tuple[int, int, int, int]:
        """Fixes the position of a rectangle to fit into the graphics scene."""
        # Fix negative bounds
        x, width = max(0, x), width + min(0, x)
        y, height = max(0, y), height + min(0, y)
        # Fix positive bounds
        if x + width > max_width:
            width = max_width - x
        if y + height > max_height:
            height = max_height - y
        # If width or height became negative, do not show the rect
        width, height = max(0, width), max(0, height)
        return x, y, width, height
