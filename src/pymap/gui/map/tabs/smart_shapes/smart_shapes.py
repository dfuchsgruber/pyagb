"""Tab for automatic shapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtOpenGLWidgets, QtWidgets
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGridLayout,
    QHBoxLayout,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from pymap.gui.history import AddOrRemoveSmartShape
from pymap.gui.icon import Icon, icon_paths
from pymap.gui.map.tabs.blocks_like import BlocksLikeTab
from pymap.gui.map.tabs.smart_shapes.shape_block_image import (
    smart_shape_get_block_image,
)
from pymap.gui.smart_shape.smart_shape import SmartShape

from .add_dialog import AddSmartShapeDialog
from .blocks import SmartShapesBlocksScene
from .edit_dialog import EditSmartShapeDialog

if TYPE_CHECKING:
    from ...map_widget import MapWidget


class SmartShapesTab(BlocksLikeTab):
    """Tab for the automatic shapes."""

    def __init__(self, map_widget: MapWidget, parent: QWidget | None = None):
        """Initialize the tab."""
        super().__init__(map_widget, parent)
        smart_shapes_layout = QVBoxLayout()
        self.setLayout(smart_shapes_layout)

        buttons_container = QWidget()
        buttons_layout = QHBoxLayout()
        buttons_container.setLayout(buttons_layout)
        smart_shapes_layout.addWidget(buttons_container)
        self.combo_box_smart_shapes = QtWidgets.QComboBox()
        self.combo_box_smart_shapes.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.combo_box_smart_shapes.currentIndexChanged.connect(
            self.load_smart_shape_and_map
        )
        buttons_layout.addWidget(self.combo_box_smart_shapes)

        self.button_add_auto_shape = QtWidgets.QPushButton('')
        self.button_add_auto_shape.setIcon(QIcon(icon_paths[Icon.PLUS]))
        self.button_add_auto_shape.clicked.connect(self.add_auto_shape)
        self.button_add_auto_shape.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_add_auto_shape.setToolTip('Add new automatic shape.')
        buttons_layout.addWidget(self.button_add_auto_shape)

        self.button_remove_auto_shape = QtWidgets.QPushButton('')
        self.button_remove_auto_shape.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        self.button_remove_auto_shape.clicked.connect(self.remove_auto_shape)
        self.button_remove_auto_shape.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_remove_auto_shape.setToolTip('Remove automatic shape.')
        buttons_layout.addWidget(self.button_remove_auto_shape)
        self.button_edit_auto_shape = QtWidgets.QPushButton('')
        self.button_edit_auto_shape.setIcon(QIcon(icon_paths[Icon.EDIT]))
        self.button_edit_auto_shape.clicked.connect(self.edit_auto_shape)
        self.button_edit_auto_shape.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_edit_auto_shape.setToolTip('Edit automatic shape.')
        buttons_layout.addWidget(self.button_edit_auto_shape)

        buttons_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        )

        self.button_reload_auto_shape = QtWidgets.QPushButton('')
        self.button_reload_auto_shape.setIcon(QIcon(icon_paths[Icon.RELOAD]))
        self.button_reload_auto_shape.clicked.connect(self.reload_auto_shape)
        self.button_reload_auto_shape.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_reload_auto_shape.setToolTip('Generate tiles from automatic shape.')
        buttons_layout.addWidget(self.button_reload_auto_shape)

        self.button_clear_auto_shape = QtWidgets.QPushButton('')
        self.button_clear_auto_shape.setIcon(QIcon(icon_paths[Icon.CLEAR]))
        self.button_clear_auto_shape.clicked.connect(self.clear_auto_shape)
        self.button_clear_auto_shape.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.button_clear_auto_shape.setToolTip('Clear automatic shape.')
        buttons_layout.addWidget(self.button_clear_auto_shape)

        tiles_layout = QGridLayout()
        smart_shapes_layout.addLayout(tiles_layout)

        self.smart_shapes_scene = QGraphicsScene()
        self.smart_shapes_scene_view = QtWidgets.QGraphicsView()
        self.smart_shapes_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.smart_shapes_scene_view.setScene(self.smart_shapes_scene)
        self.smart_shapes_scene_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        tiles_layout.addWidget(self.smart_shapes_scene_view, 1, 2, 2, 1)

        group_selection = QtWidgets.QGroupBox('Selection')
        tiles_layout.addWidget(group_selection, 1, 1, 1, 1)
        self.selection_scene = QGraphicsScene()
        self.selection_scene_view = QtWidgets.QGraphicsView()
        self.selection_scene_view.setScene(self.selection_scene)
        self.selection_scene_view.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.selection_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        selection_scene_layout = QVBoxLayout()
        group_selection.setLayout(selection_scene_layout)
        selection_scene_layout.addWidget(self.selection_scene_view)

        self.blocks_scene = SmartShapesBlocksScene(self)
        self.blocks_scene_view = QtWidgets.QGraphicsView()
        self.blocks_scene_view.setScene(self.blocks_scene)
        self.blocks_scene_view.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.blocks_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        tiles_layout.addWidget(self.blocks_scene_view, 2, 1, 1, 1)

        group_properties = QtWidgets.QGroupBox('Properties')
        group_properties.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding
        )
        self.properties_layout = QVBoxLayout()
        group_properties.setLayout(self.properties_layout)
        smart_shapes_layout.addWidget(group_properties)
        smart_shapes_layout.addStretch()

    @property
    def current_smart_shape_name(self) -> str:
        """The name of the current smart shape."""
        return self.combo_box_smart_shapes.currentText()

    @property
    def current_smart_shape(self) -> SmartShape | None:
        """The current smart shape."""
        if not self.current_smart_shape_name:
            return None
        assert self.current_smart_shape_name in self.map_widget.main_gui.smart_shapes, (
            f'Smart shape {self.current_smart_shape_name} not found'
        )
        return self.map_widget.main_gui.smart_shapes[self.current_smart_shape_name]

    @property
    def smart_shape_blocks(self) -> NDArray[np.int_]:
        """The blocks of the smart shape. Note that these are not border padded."""
        return self.map_widget.main_gui.smart_shapes[
            self.current_smart_shape_name
        ].buffer

    def add_auto_shape(self):
        """Add an automatic shape."""
        dialog = AddSmartShapeDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            name, smart_shape = dialog.get_smart_shape()
            assert name not in self.map_widget.main_gui.smart_shapes, (
                f'Smart shape {name} already exists.'
            )
            self.map_widget.undo_stack.push(
                AddOrRemoveSmartShape(self, name, smart_shape, True)
            )

    def remove_auto_shape(self):
        """Remove the automatic shape."""
        if self.combo_box_smart_shapes.currentIndex() == -1:
            return
        name = self.combo_box_smart_shapes.currentText()
        assert name in self.map_widget.main_gui.smart_shapes, (
            f'Smart shape {name} not found.'
        )
        self.map_widget.undo_stack.push(
            AddOrRemoveSmartShape(
                self,
                name,
                self.map_widget.main_gui.smart_shapes[name],
                False,
            )
        )

    def reload_auto_shape(self):
        """Reload the automatic shape, i.e. re-generates it from the metatile map."""
        smart_shape = self.current_smart_shape
        if smart_shape is None:
            return
        assert self.map_widget.main_gui.project is not None, 'Project is not loaded'
        template = self.map_widget.main_gui.project.smart_shape_templates[
            smart_shape.template
        ]
        blocks, mask = template.generate_blocks(
            smart_shape,
            self.map_widget.main_gui.get_map_blocks(),
        )
        self.map_widget.main_gui.replace_blocks_at(mask, 0, blocks)

    def clear_auto_shape(self):
        """Clear the automatic shape."""
        self.map_widget.main_gui.smart_shape_clear(self.current_smart_shape_name)

    def edit_auto_shape(self):
        """Edit the automatic shape."""
        if self.smart_shape_loaded:
            dialog = EditSmartShapeDialog(
                self, self.combo_box_smart_shapes.currentText()
            )
            dialog.exec()

    @property
    def connectivity_layer(self) -> int:
        """The layer for connectivity.

        Use for flood filling, replacement, etc.
        """
        return 0

    @property
    def selected_layers(self) -> NDArray[np.int_]:
        """The selected layers."""
        return np.array([0, 1])

    def load_project(self) -> None:
        """Loads the project."""
        super().load_project()

    def load_header(self):
        """Loads the tab."""
        self.button_add_auto_shape.setEnabled(self.map_widget.header_loaded)
        self.update_smart_shapes()
        self.load_smart_shape()
        super().load_header()

    def load_smart_shape_and_map(self):
        """Reloads the smart shape and updates the map with the smart shape."""
        self.map_widget.load_map()

    @property
    def blocks(self) -> NDArray[np.int_] | None:
        """The blocks."""
        smart_shape = self.current_smart_shape
        if smart_shape is None:
            return None
        else:
            # Add padding to the blocks by creating a copy
            # This is inefficient...
            padded_width, padded_height = self.map_widget.main_gui.get_border_padding()
            map_width, map_height = self.map_widget.main_gui.get_map_dimensions()
            blocks = np.zeros(
                (
                    map_height + 2 * padded_height,
                    map_width + 2 * padded_width,
                    *smart_shape.buffer.shape[2:],
                ),
                dtype=int,
            )
            blocks[
                padded_height : padded_height + map_height,
                padded_width : padded_width + map_width,
            ] = smart_shape.buffer
            return blocks

    def set_selection(self, selection: NDArray[np.int_]) -> None:
        """Sets the selection.

        Args:
            selection (NDArray[np.int_]): The selection.
        """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        if (
            not self.map_widget.main_gui.footer_loaded
            or self.current_smart_shape is None
        ):
            return
        assert self.map_widget.main_gui.project is not None, 'Project is not loaded'
        template = self.map_widget.main_gui.project.smart_shape_templates[
            self.current_smart_shape.template
        ]

        for (y, x), block in np.ndenumerate(selection[:, :, 0]):
            item = QGraphicsPixmapItem(template.block_pixmaps[block])
            self.selection_scene.addItem(item)
            item.setPos(16 * x, 16 * y)  # 16 pixels per block

        self.selection_scene.setSceneRect(
            0,
            0,
            selection.shape[1] * 16,
            selection.shape[0] * 16,  # 16 pixels per block
        )

    def set_blocks_at(
        self, x: int, y: int, layers: NDArray[np.int_], blocks: NDArray[np.int_]
    ):
        """Sets the blocks at the given position.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            layers (NDArray[np.int_]): The layers.
            blocks (NDArray[np.int_]): The blocks.
        """
        self.map_widget.main_gui.smart_shape_set_blocks_at(
            self.current_smart_shape_name, x, y, blocks
        )

    def replace_blocks(self, x: int, y: int, layer: int, block: NDArray[np.int_]):
        """Replaces the blocks.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            layer (int): The layer.
            block (NDArray[np.int_]): The block.
        """
        self.map_widget.main_gui.smart_shape_replace_blocks(
            self.current_smart_shape_name, x, y, layer, block
        )

    def flood_fill(self, x: int, y: int, layer: int, block: NDArray[np.int_]):
        """Flood fills the blocks.

        Args:
            x (int): The x coordinate.
            y (int): The y coordinate.
            layer (int): The layer.
            block (NDArray[np.int_]): The block.
        """
        self.map_widget.main_gui.smart_shape_flood_fill(
            self.current_smart_shape_name,
            x,
            y,
            block,
            layer=self.connectivity_layer,
        )

    @property
    def smart_shape_loaded(self) -> bool:
        """Returns whether the smart shape is loaded."""
        return (
            self.map_widget.header_loaded
            and self.combo_box_smart_shapes.currentIndex() != -1
        )

    def update_smart_shapes(self, load_index: str | None = None):
        """Updates the list of smart shapes.

        Args:
            load_index (str | None): The name (of the shape) to load, if any.
        """
        self.combo_box_smart_shapes.blockSignals(True)
        self.combo_box_smart_shapes.clear()
        if self.map_widget.header_loaded:
            for name in self.map_widget.main_gui.smart_shapes:
                self.combo_box_smart_shapes.addItem(name)

        # Trigger the update from setting an index in the combo box
        if self.combo_box_smart_shapes.count() == 0:
            self.combo_box_smart_shapes.setCurrentIndex(-1)
        if load_index is not None:
            index = self.combo_box_smart_shapes.findText(load_index)
            if index != -1:
                self.combo_box_smart_shapes.setCurrentIndex(index)
        self.load_smart_shape()
        self.combo_box_smart_shapes.blockSignals(False)

    def _update_smart_shape_enabled(self):
        """Update the smart shape."""
        self.button_remove_auto_shape.setEnabled(self.smart_shape_loaded)
        self.button_edit_auto_shape.setEnabled(self.smart_shape_loaded)
        self.button_reload_auto_shape.setEnabled(self.smart_shape_loaded)
        self.button_clear_auto_shape.setEnabled(self.smart_shape_loaded)
        self.combo_box_smart_shapes.setEnabled(self.smart_shape_loaded)
        self.smart_shapes_scene_view.setEnabled(self.smart_shape_loaded)
        self.selection_scene_view.setEnabled(self.smart_shape_loaded)

    def load_smart_shape(self):
        """Update the smart shape."""
        self.selection_scene.clear()
        self._clear_properties_layout()
        self._update_smart_shape_enabled()
        self.update_smart_shapes_scene()
        self.blocks_scene.load_smart_shape()

    def update_smart_shapes_scene(self):
        """Updates the smart shapes scene."""
        self.smart_shapes_scene.clear()
        if self.current_smart_shape is not None:
            assert self.map_widget.main_gui.block_images is not None
            assert self.map_widget.main_gui.project is not None, 'Project is not loaded'
            template = self.map_widget.main_gui.project.smart_shape_templates[
                self.current_smart_shape.template
            ]

            self.smart_shapes_scene.addPixmap(template.template_pixmap)
            for _, pixmap_item in np.ndenumerate(
                smart_shape_get_block_image(
                    self.current_smart_shape,
                    self.map_widget.main_gui.block_images,
                )
            ):
                pixmap_item.setAcceptHoverEvents(True)
                self.smart_shapes_scene.addItem(pixmap_item)

    def _clear_properties_layout(self):
        """Clear the properties layout."""
        while self.properties_layout.count() > 0:
            item = self.properties_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def load_map(self):
        """Loads the map image."""
        self.map_widget.add_block_images_to_scene()
        self.map_widget.add_smart_shape_images_to_scene()
