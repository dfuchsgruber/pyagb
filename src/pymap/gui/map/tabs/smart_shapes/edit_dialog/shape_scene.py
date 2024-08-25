"""Widget where the shape is drawn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QGraphicsScene, QGraphicsSceneMouseEvent, QWidget

if TYPE_CHECKING:
    from .edit_dialog import EditSmartShapeDialog


class ShapeScene(QGraphicsScene):
    """Scene where the shape is drawn."""

    def __init__(
        self, edit_dialog: EditSmartShapeDialog, parent: QWidget | None = None
    ):
        """Initializes the shape scene.

        Args:
            edit_dialog (EditSmartShapeDialog): The edit dialog.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.edit_dialog = edit_dialog

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns the dimensions of the scene.

        Returns:
            tuple[int, int]: The dimensions.
        """
        smart_shape = self.edit_dialog.smart_shapes_tab.current_smart_shape
        assert smart_shape is not None, 'Smart shape is not loaded'
        assert self.edit_dialog.main_gui.project is not None, 'Project is not loaded'
        template = self.edit_dialog.main_gui.project.smart_shape_templates[
            smart_shape.template
        ]
        return template.dimensions

    def load_template(self, template_name: str) -> None:
        """Loads the template.

        Args:
            template_name (str): The template.
        """
        assert self.edit_dialog.main_gui.project is not None, 'Project is not loaded'
        template = self.edit_dialog.main_gui.project.smart_shape_templates[
            template_name
        ]
        self.template_pixmap_item = self.addPixmap(template.template_pixmap)

    def load_shape(self, shape: str) -> None:
        """Loads the shape.

        Args:
            shape (str): The shape.
        """
        assert self.edit_dialog.main_gui.project is not None, 'Project is not loaded'
        smart_shape = self.edit_dialog.main_gui.smart_shapes[shape]
        self.load_template(smart_shape.template)

    def clear(self) -> None:
        """Clears the scene."""
        super().clear()
        self.template_pixmap_item = None

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        """Mouse press event.

        Args:
            event (QGraphicsSceneMouseEvent): The mouse event.
        """
        if not self.edit_dialog.main_gui.footer_loaded:
            return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x not in range(self.dimensions[0]) or y not in range(self.dimensions[1]):
            return
        self.edit_dialog.set_shape_blocks(
            x, y, self.edit_dialog.selection[..., 0]
        )  # For now, smart shapes have no depth (i.e. level information)
