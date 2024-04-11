"""Tab for automatic shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtOpenGLWidgets, QtWidgets
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from .auto_shapes_scene import AutoShapesScene
from .tab import MapWidgetTab

if TYPE_CHECKING:
    from ..map_widget import MapWidget


class AutoShapesTab(MapWidgetTab):
    """Tab for the automatic shapes."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)
        auto_shapes_widget = QWidget()
        auto_shapes_layout = QVBoxLayout()
        auto_shapes_widget.setLayout(auto_shapes_layout)

        self.auto_shapes_scene = AutoShapesScene(self)
        self.auto_shapes_scene_view = QtWidgets.QGraphicsView()
        self.auto_shapes_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.auto_shapes_scene_view.setScene(self.auto_shapes_scene)
        auto_shapes_layout.addWidget(self.auto_shapes_scene_view)

    @property
    def selected_layers(self) -> NDArray[np.int_]:
        """Get the selected levels."""
        return np.array([0])

    def load_project(self) -> None:
        """Loads the project."""
        ...

    def load_header(self):
        """Loads the tab."""
        ...
