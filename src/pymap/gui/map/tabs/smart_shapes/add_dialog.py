"""Create dialog for smart shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QLabel,
    QLineEdit,
)

from pymap.gui.smart_shape.smart_shape import SmartShape

if TYPE_CHECKING:
    from .smart_shapes import SmartShapesTab


class AddSmartShapeDialog(QDialog):
    """Dialog for creating smart shapes."""

    def __init__(self, parent: SmartShapesTab):
        """Initialize the dialog.

        Args:
            parent (SmartShapesTab): The parent tab.
        """
        super().__init__(parent=parent)
        self.setWindowTitle('Create Smart Shape')
        layout = QGridLayout()
        self.setLayout(layout)

        self.smart_shapes_tab = parent
        layout.addWidget(QLabel('Template'), 2, 1)
        self.combo_box_smart_shapes = QComboBox()
        layout.addWidget(self.combo_box_smart_shapes, 2, 2)
        assert (
            self.smart_shapes_tab.map_widget.main_gui.project is not None
        ), 'A project must be loaded.'
        self.combo_box_smart_shapes.addItems(
            list(
                self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates
            )
        )
        layout.addWidget(QLabel('Name'), 1, 1)
        self.name_line_edit = QLineEdit()
        layout.addWidget(self.name_line_edit, 1, 2)

        self.errorLabel = QLabel('')
        self.errorLabel.setStyleSheet('color: red; font-size: small;')
        layout.addWidget(self.errorLabel, 3, 1, 1, 2)

        self.preview_scene = QGraphicsScene()
        self.preview_scene_view = QGraphicsView()
        self.preview_scene_view.setViewport(QOpenGLWidget())
        self.preview_scene_view.setScene(self.preview_scene)
        layout.addWidget(self.preview_scene_view, 4, 1, 1, 2)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept_if_valid)
        self.button_box.rejected.connect(self.reject)
        self.name_line_edit.textChanged.connect(self.validate_name)
        self.combo_box_smart_shapes.currentIndexChanged.connect(self.update_preview)
        layout.addWidget(self.button_box, 5, 1, 1, 2)

        self.validate_name()
        self.update_preview()

    def update_preview(self):
        """Update the preview."""
        assert (
            self.smart_shapes_tab.map_widget.main_gui.project is not None
        ), 'A project must be loaded.'
        template = (
            self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates[
                self.combo_box_smart_shapes.currentText()
            ]
        )
        self.preview_scene.clear()
        self.preview_scene.addPixmap(template.template_pixmap)

    def name_is_valid(self) -> bool:
        """Checks that none of the smart shapes have the same name as the one entered."""
        assert (
            self.smart_shapes_tab.map_widget.main_gui.project is not None
        ), 'A project must be loaded.'
        return (
            self.name_line_edit.text()
            not in self.smart_shapes_tab.map_widget.main_gui.smart_shapes.keys()
        )

    def validate_name(self) -> bool:
        """Updates the error label if the name is invalid."""
        if self.name_is_valid():
            self.errorLabel.setText('')
            return True
        else:
            self.errorLabel.setText('Smart Shape already exists.')
            return False

    def accept_if_valid(self):
        """Accept the dialog."""
        if self.validate_name():
            return super().accept()
        else:
            self.name_line_edit.setFocus()

    def get_smart_shape(self) -> tuple[str, SmartShape]:
        """Get the smart shape."""
        assert (
            self.smart_shapes_tab.map_widget.main_gui.project is not None
        ), 'A project must be loaded.'
        template = (
            self.smart_shapes_tab.map_widget.main_gui.project.smart_shape_templates[
                self.combo_box_smart_shapes.currentText()
            ]
        )
        width, height = template.dimensions
        blocks: list[list[int]] = np.zeros((height, width), dtype=int).tolist()
        width, height = self.smart_shapes_tab.map_widget.main_gui.get_map_dimensions()
        shape: SmartShape = SmartShape(
            self.combo_box_smart_shapes.currentText(), blocks, height, width
        )
        return self.name_line_edit.text(), shape
