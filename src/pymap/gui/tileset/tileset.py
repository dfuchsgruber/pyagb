"""Widget for tileset editing."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from PySide6 import QtGui, QtOpenGLWidgets, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import QFileDialog, QGraphicsPixmapItem, QMessageBox, QSizePolicy

from agb import image as agbimage
from agb.model.type import ModelValue
from pymap.gui.render import ndarray_to_QImage
from pymap.gui.tileset.block_properties import BlockProperties
from pymap.gui.tileset.tileset_properties import TilesetProperties

from .. import history, properties, render
from .block_scene import BlockScene
from .blocks_scene import BlocksScene
from .selection_scene import SelectionScene
from .tiles_scene import TileFlip, TilesScene

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class TilesetWidget(QtWidgets.QWidget):
    """Widget for editing tilesets."""

    def __init__(self, main_gui: PymapGui, parent: QtWidgets.QWidget | None = None):
        """Widget for editing tilesets.

        Args:
            main_gui (PymapGui): Reference to the main gui.
            parent (QtWidgets.QWidget | None, optional): Parent. Defaults to None.
        """
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.undo_stack = QtGui.QUndoStack()
        self.undo_stack.canRedoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.canUndoChanged.connect(self._update_undo_redo_tooltips)
        self.undo_stack.indexChanged.connect(self._update_undo_redo_tooltips)

        self.behaviour_clipboard = None
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        self.zoom_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal, self)
        self.zoom_slider.setMinimum(5)
        self.zoom_slider.setMaximum(40)
        self.zoom_slider.setTickInterval(1)

        self.horizontal_splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        self.horizontal_splitter.restoreState(
            self.main_gui.settings.value('tileset/splitter/horizontal', b'', bytes)  # type: ignore
        )

        blocks_group = QtWidgets.QGroupBox('Blocks')
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QtWidgets.QGraphicsView()
        self.blocks_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        blocks_layout = QtWidgets.QGridLayout()
        blocks_group.setLayout(blocks_layout)
        blocks_layout.addWidget(self.blocks_scene_view)
        self.horizontal_splitter.addWidget(blocks_group)
        self.horizontal_splitter.splitterMoved.connect(
            lambda: self.main_gui.settings.setValue(
                'tileset/splitter/horizontal', self.horizontal_splitter.saveState()
            )
        )

        self.gfx_column = QtWidgets.QFrame()
        self.gfx_column_layout = QtWidgets.QVBoxLayout()
        self.gfx_column.setLayout(self.gfx_column_layout)

        gfx_group = QtWidgets.QGroupBox('Gfx')
        gfx_layout = QtWidgets.QGridLayout()
        gfx_group.setLayout(gfx_layout)
        gfx_layout.addWidget(QtWidgets.QLabel('Primary'), 1, 1, 1, 1)
        self.gfx_primary_combobox = QtWidgets.QComboBox()
        gfx_layout.addWidget(self.gfx_primary_combobox, 1, 2, 1, 1)
        gfx_layout.addWidget(QtWidgets.QLabel('Secondary'), 2, 1, 1, 1)
        self.gfx_secondary_combobox = QtWidgets.QComboBox()
        gfx_layout.addWidget(self.gfx_secondary_combobox, 2, 2, 1, 1)

        self.gfx_primary_combobox.currentTextChanged.connect(
            partial(self.main_gui.change_gfx, primary=True)
        )
        self.gfx_secondary_combobox.currentTextChanged.connect(
            partial(self.main_gui.change_gfx, primary=False)
        )
        gfx_layout.setColumnStretch(1, 0)
        gfx_layout.setColumnStretch(2, 1)
        self.gfx_column_layout.addWidget(gfx_group, 1)

        properties_group = QtWidgets.QGroupBox('Properties')
        properties_layout = QtWidgets.QGridLayout()
        properties_group.setLayout(properties_layout)
        self.properties_tree_tsp = TilesetProperties(self, True)
        properties_layout.addWidget(self.properties_tree_tsp, 0, 0, 1, 1)
        self.properties_tree_tss = TilesetProperties(self, False)
        properties_layout.addWidget(self.properties_tree_tss, 0, 1, 1, 1)
        self.gfx_column_layout.addWidget(properties_group, 1)

        seleciton_group = QtWidgets.QGroupBox('Selection')
        selection_layout = QtWidgets.QGridLayout()
        seleciton_group.setLayout(selection_layout)
        self.selection_scene = SelectionScene(self)
        self.selection_scene_view = QtWidgets.QGraphicsView()
        self.selection_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.selection_scene_view.setScene(self.selection_scene)
        selection_layout.addWidget(self.selection_scene_view)
        self.gfx_column_layout.addWidget(seleciton_group, 1)

        current_block_group = QtWidgets.QGroupBox('Current Block')
        current_block_layout = QtWidgets.QGridLayout()
        current_block_group.setLayout(current_block_layout)
        lower_group = QtWidgets.QGroupBox('Bottom Layer')
        lower_layout = QtWidgets.QGridLayout()
        lower_group.setLayout(lower_layout)
        self.block_lower_scene = BlockScene(self, 0)
        self.block_lower_scene_view = QtWidgets.QGraphicsView()
        self.block_lower_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.block_lower_scene_view.setScene(self.block_lower_scene)
        lower_layout.addWidget(self.block_lower_scene_view)
        mid_group = QtWidgets.QGroupBox('Mid Layer')
        mid_layout = QtWidgets.QGridLayout()
        mid_group.setLayout(mid_layout)
        self.block_mid_scene = BlockScene(self, 1)
        self.block_mid_scene_view = QtWidgets.QGraphicsView()
        self.block_mid_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.block_mid_scene_view.setScene(self.block_mid_scene)
        mid_layout.addWidget(self.block_mid_scene_view)
        upper_group = QtWidgets.QGroupBox('Upper Layer')
        upper_layout = QtWidgets.QGridLayout()
        upper_group.setLayout(upper_layout)
        self.block_upper_scene = BlockScene(self, 2)
        self.block_upper_scene_view = QtWidgets.QGraphicsView()
        self.block_upper_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.block_upper_scene_view.setScene(self.block_upper_scene)
        upper_layout.addWidget(self.block_upper_scene_view)
        current_block_layout.addWidget(lower_group, 1, 1, 1, 1)
        current_block_layout.addWidget(mid_group, 1, 2, 1, 1)
        current_block_layout.addWidget(upper_group, 1, 3, 1, 1)
        self.block_properties = BlockProperties(self)
        current_block_layout.addWidget(self.block_properties, 2, 1, 1, 3)
        behaviour_clipboard_layout = QtWidgets.QGridLayout()
        current_block_layout.addLayout(behaviour_clipboard_layout, 3, 1, 1, 2)
        self.behaviour_clipboard_copy = QtWidgets.QPushButton('Copy')
        self.behaviour_clipboard_copy.clicked.connect(self.copy_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_copy, 1, 1)
        self.behaviour_clipboard_paste = QtWidgets.QPushButton('Paste')
        self.behaviour_clipboard_paste.clicked.connect(self.paste_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_paste, 1, 2)
        self.behaviour_clipboard_clear = QtWidgets.QPushButton('Clear')
        self.behaviour_clipboard_clear.clicked.connect(self.clear_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_clear, 1, 3)
        behaviour_clipboard_layout.addWidget(QtWidgets.QLabel(), 1, 4)

        current_block_layout.setRowStretch(1, 0)
        current_block_layout.setRowStretch(2, 1)
        current_block_layout.setRowStretch(3, 0)

        self.gfx_column_layout.addWidget(current_block_group, 1)
        self.horizontal_splitter.addWidget(self.gfx_column)

        tiles_group = QtWidgets.QGroupBox('Tiles')
        tiles_layout = QtWidgets.QGridLayout()
        tiles_group.setLayout(tiles_layout)

        self.tiles_mirror_horizontal_checkbox = QtWidgets.QCheckBox('H-Flip')
        tiles_layout.addWidget(self.tiles_mirror_horizontal_checkbox, 1, 2, 1, 1)
        self.tiles_mirror_horizontal_checkbox.toggled.connect(self.update_tiles)
        self.tiles_mirror_vertical_checkbox = QtWidgets.QCheckBox('V-Flip')
        tiles_layout.addWidget(self.tiles_mirror_vertical_checkbox, 1, 3, 1, 1)
        self.tiles_mirror_vertical_checkbox.toggled.connect(self.update_tiles)
        tiles_palette_group = QtWidgets.QGroupBox('Palette')
        tiles_palette_group_layout = QtWidgets.QGridLayout()
        tiles_palette_group.setLayout(tiles_palette_group_layout)
        self.tiles_palette_combobox = QtWidgets.QComboBox()
        self.tiles_palette_combobox.addItems(list(map(str, range(13))))
        self.tiles_palette_combobox.currentIndexChanged.connect(self.update_tiles)
        tiles_palette_group_layout.addWidget(self.tiles_palette_combobox, 1, 1, 1, 1)
        tiles_import_button = QtWidgets.QPushButton('Import')
        tiles_import_button.clicked.connect(self.import_palette)
        tiles_palette_group_layout.addWidget(tiles_import_button, 1, 2, 1, 1)
        tiles_export_button = QtWidgets.QPushButton('Export')
        tiles_export_button.clicked.connect(self.export_gfx)
        tiles_palette_group_layout.addWidget(tiles_export_button, 1, 3, 1, 1)
        tiles_palette_group_layout.setColumnStretch(1, 0)
        tiles_palette_group_layout.setColumnStretch(2, 0)
        tiles_palette_group_layout.setColumnStretch(3, 0)

        tiles_layout.addWidget(tiles_palette_group, 1, 1, 1, 1)
        self.tiles_scene = TilesScene(self)
        self.tiles_scene_view = QtWidgets.QGraphicsView()
        self.tiles_scene_view.setViewport(QtOpenGLWidgets.QOpenGLWidget())
        self.tiles_scene_view.setScene(self.tiles_scene)
        tiles_layout.addWidget(self.tiles_scene_view, 3, 1, 1, 3)
        tiles_layout.setColumnStretch(1, 1)
        tiles_layout.setColumnStretch(2, 0)
        tiles_layout.setColumnStretch(3, 0)
        self.horizontal_splitter.addWidget(tiles_group)

        zoom_group = QtWidgets.QGroupBox('Zoom')
        zoom_layout = QtWidgets.QGridLayout()
        zoom_group.setLayout(zoom_layout)
        zoom_layout.addWidget(self.zoom_slider, 1, 1, 1, 1)
        self.zoom_label = QtWidgets.QLabel()
        zoom_layout.addWidget(self.zoom_label, 1, 2, 1, 1)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        self.zoom_slider.setValue(
            self.main_gui.settings.value('tileset/zoom', 10, int)  # type: ignore
        )

        self.info_label = QtWidgets.QLabel('')
        self.info_label.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )

        layout.addWidget(zoom_group)
        self.horizontal_splitter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.horizontal_splitter)
        layout.addWidget(self.info_label)

        self.gfx_column_layout.setStretch(0, 0)
        self.gfx_column_layout.setStretch(1, 1)
        self.gfx_column_layout.setStretch(2, 0)
        self.gfx_column_layout.setStretch(3, 3)

        self.load_project()  # Initialize widgets as disabled

    @property
    def tiles_flip(self) -> TileFlip:
        """The current flip of the tiles.

        Returns:
            TileFlip: The current flip of the tiles.
        """
        flip = 0
        if self.tiles_mirror_horizontal_checkbox.isChecked():
            flip |= TileFlip.HORIZONTAL
        if self.tiles_mirror_vertical_checkbox.isChecked():
            flip |= TileFlip.VERTICAL
        return TileFlip(flip)

    @property
    def tileset_loaded(self) -> bool:
        """If a tileset is currently loaded."""
        return (
            self.main_gui.project is not None
            and self.main_gui.header is not None
            and self.main_gui.footer is not None
            and self.main_gui.tileset_primary is not None
            and self.main_gui.tileset_secondary is not None
        )

    def _update_undo_redo_tooltips(
        self,
    ):
        """Updates the undo and redo tooltips."""
        self.main_gui.update_redo_undo_tooltips(
            self,
            self.undo_stack,
        )

    def load_project(self):
        """Loads a new project."""
        if self.main_gui.project is None:
            return
        self.gfx_primary_combobox.blockSignals(True)
        self.gfx_primary_combobox.clear()
        self.gfx_primary_combobox.addItems(
            list(self.main_gui.project.gfxs_primary.keys())
        )
        self.gfx_primary_combobox.blockSignals(False)
        self.gfx_secondary_combobox.blockSignals(True)
        self.gfx_secondary_combobox.clear()
        self.gfx_secondary_combobox.addItems(
            list(self.main_gui.project.gfxs_secondary.keys())
        )
        self.gfx_secondary_combobox.blockSignals(False)
        self.load_header()

    def load_header(self):
        """Updates the blocks of a new header."""
        self.tiles_scene.clear()
        self.blocks_scene.clear()
        self.selection_scene.clear()
        self.blocks_scene.clear()
        self.selected_block_idx = 0
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            self.behaviour_clipboard_paste.setEnabled(False)
            self.behaviour_clipboard_copy.setEnabled(False)
            self.behaviour_clipboard_clear.setEnabled(False)
            self.gfx_primary_combobox.setEnabled(False)
            self.gfx_secondary_combobox.setEnabled(False)
            self.properties_tree_tsp.setEnabled(False)  # type: ignore
            self.properties_tree_tss.setEnabled(False)  # type: ignore
        else:
            self.behaviour_clipboard_copy.setEnabled(True)
            self.behaviour_clipboard_clear.setEnabled(True)
            self.gfx_primary_combobox.setEnabled(True)
            self.gfx_secondary_combobox.setEnabled(True)
            gfx_primary_label = self.main_gui.get_tileset_gfx_label(True)
            gfx_secondary_label = self.main_gui.get_tileset_gfx_label(False)
            assert isinstance(gfx_primary_label, str)
            self.gfx_primary_combobox.blockSignals(True)
            self.gfx_primary_combobox.setCurrentText(gfx_primary_label)
            self.gfx_primary_combobox.blockSignals(False)
            assert isinstance(gfx_secondary_label, str)
            self.gfx_secondary_combobox.blockSignals(True)
            self.gfx_secondary_combobox.setCurrentText(gfx_secondary_label)
            self.gfx_secondary_combobox.blockSignals(False)

            self.properties_tree_tsp.setEnabled(True)  # type: ignore
            self.properties_tree_tss.setEnabled(True)  # type: ignore
            self.properties_tree_tsp.load()
            self.properties_tree_tss.load()

            self.load_tiles()
            self.load_blocks()
            self.set_selection(np.array([[self.get_empty_block_tile()]]))
            self.set_current_block(0)

    def get_empty_block_tile(self) -> ModelValue:
        """Creates an empty block tile.

        Returns:
            ModelValue: The empty block tile.
        """
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        config = self.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]
        datatype = self.main_gui.project.model[config['block_datatype']]
        tileset = (
            self.main_gui.tileset_primary
            if self.selected_block_idx < 0x280
            else self.main_gui.tileset_secondary
        )
        # Create a new "empty" instance
        return datatype(
            self.main_gui.project,
            config['blocks_path'] + [self.selected_block_idx % 0x280, 0],
            properties.get_parents_by_path(
                tileset, config['blocks_path'] + [self.selected_block_idx % 0x280, 0]
            ),
        )

    @property
    def selected_block(self) -> npt.NDArray[np.object_]:
        """Gets the data of the currently selected block.

        Returns:
            RGBAImage: The data of the currently selected block.
        """
        return self.main_gui.get_block(self.selected_block_idx)

    def reload(self):
        """Reloads the entire view (in case tiles or gfx have changed)."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.load_tiles()
        self.load_blocks()
        self.set_current_block(self.selected_block_idx)
        self.set_selection(self.selection)

    # @ProfileBlock('load_tiles')
    def load_tiles(self):
        """Reloads the tiles."""
        if (
            self.main_gui.project is None
            or self.main_gui.tiles is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.tile_pixmaps: list[dict[int, QPixmap]] = []
        # Convert masks to ints for efficiency
        mask_horizontal = int(TileFlip.HORIZONTAL)
        mask_vertical = int(TileFlip.VERTICAL)

        for palette_idx in range(13):
            pixmaps: dict[int, QPixmap] = {}
            for flip in range(4):
                # Assemble the entire picture
                image = np.zeros((512, 128, 4), dtype=np.uint8)
                for idx, tile_img in enumerate(self.main_gui.tiles[palette_idx]):
                    x, y = idx % 16, idx // 16
                    if flip & mask_horizontal:
                        tile_img = tile_img[:, ::-1]
                        x = 15 - x
                    if flip & mask_vertical:
                        tile_img = tile_img[::-1]
                        y = 63 - y
                    image[8 * y : 8 * (y + 1), 8 * x : 8 * (x + 1)] = tile_img
                pixmaps[flip] = QPixmap.fromImage(ndarray_to_QImage(image))
            self.tile_pixmaps.append(pixmaps)
        self.update_tiles()

    def load_blocks(self):
        """Reloads the blocks."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        assert self.main_gui.block_images is not None
        self.blocks_image = render.draw_blocks_pool(self.main_gui.block_images)
        self.update_blocks()

    def set_current_block(self, block_idx: int):
        """Reloads the current block."""
        self.block_lower_scene.clear()
        self.block_mid_scene.clear()
        self.block_upper_scene.clear()
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.selected_block_idx = block_idx
        self.block_lower_scene.update_block()
        self.block_mid_scene.update_block()
        self.block_upper_scene.update_block()
        self.blocks_scene.update_selection_rect()
        self.block_properties.load()

    def update_blocks(self):
        """Updates the display of the blocks."""
        self.blocks_scene.clear()
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        height, width = self.blocks_image.shape[:2]
        width, height = (
            int(width * self.zoom_slider.value() / 10),
            int(height * self.zoom_slider.value() / 10),
        )
        pixmap = QPixmap.fromImage(ndarray_to_QImage(self.blocks_image)).scaled(
            width, height
        )
        item = QGraphicsPixmapItem(pixmap)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        # Add the selection rectangle
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        self.blocks_scene.selection_rect = self.blocks_scene.addRect(
            0,
            0,
            0,
            0,
            pen=QPen(color, 1.0 * self.zoom_slider.value() / 10),
            brush=Qt.BrushStyle.NoBrush,
        )
        self.blocks_scene.update_selection_rect()

    def update_tiles(self):
        """Updates the display of the tiles widget."""
        self.tiles_scene.clear()
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        width, height = (
            int(128 * self.zoom_slider.value() / 10),
            int(512 * self.zoom_slider.value() / 10),
        )
        flip = (
            TileFlip.HORIZONTAL
            if self.tiles_mirror_horizontal_checkbox.isChecked()
            else 0
        ) | (
            TileFlip.VERTICAL if self.tiles_mirror_vertical_checkbox.isChecked() else 0
        )
        item = QGraphicsPixmapItem(
            self.tile_pixmaps[self.tiles_palette_combobox.currentIndex()][flip].scaled(
                width, height
            )
        )
        self.tiles_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        # Add the selection rectangle
        self.tiles_scene.add_selection_rect()
        self.tiles_scene.update_selection_rect()
        self.tiles_scene.setSceneRect(0, 0, width, height)

    def zoom_changed(self):
        """Event handler for when the zoom has changed."""
        self.zoom_label.setText(f'{self.zoom_slider.value() * 10}%')
        self.main_gui.settings.setValue('tileset/zoom', self.zoom_slider.value())
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.update_tiles()
        self.update_blocks()
        self.set_selection(self.selection)
        self.block_lower_scene.update_block()
        self.block_mid_scene.update_block()
        self.block_upper_scene.update_block()

    def set_selection(self, selection: npt.NDArray[np.object_]):
        """Sets the selection to a set of tiles."""
        assert self.main_gui.tiles is not None
        self.selection = selection
        self.selection_scene.clear()
        image = np.zeros(
            (8 * selection.shape[0], 8 * selection.shape[1], 4), dtype=np.uint8
        )
        for (y, x), tile in np.ndenumerate(selection):
            assert isinstance(tile, dict)
            palette_idx, tile_idx = tile['palette_idx'], tile['tile_idx']  # type: ignore
            assert isinstance(palette_idx, int)
            assert isinstance(tile_idx, int)
            tile_img = self.main_gui.tiles[palette_idx, tile_idx]
            if tile['horizontal_flip']:
                tile_img = tile_img[:, ::-1]
            if tile['vertical_flip']:
                tile_img = tile_img[::-1]
            image[8 * y : 8 * (y + 1), 8 * x : 8 * (x + 1)] = tile_img
        width, height = (
            int(8 * selection.shape[1] * self.zoom_slider.value() / 10),
            int(8 * selection.shape[0] * self.zoom_slider.value() / 10),
        )
        item = QGraphicsPixmapItem(
            QPixmap.fromImage(QImage(ndarray_to_QImage(image))).scaled(width, height)
        )
        self.selection_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        self.selection_scene.setSceneRect(0, 0, width, height)

    def set_info(self, tile_idx: int | None):
        """Updates the info to a specific tile index or clears it if None is given."""
        if tile_idx is None:
            self.info_label.setText('')
        else:
            section = (
                'Primary Tileset'
                if tile_idx < 640
                else ('Secondary Tileset' if tile_idx < 1020 else 'Reserved for Doors')
            )
            self.info_label.setText(f'Tile {hex(tile_idx)}, {section}')

    def export_gfx(self):
        """Prompts the user to export to export a gfx in the selected palette."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        message_box = QMessageBox(self)
        message_box.setWindowTitle(
            f'Export Gfx in Palette {self.tiles_palette_combobox.currentIndex()}'
        )
        message_box.setText(
            f'Select which gfx to export in palette '
            f'{self.tiles_palette_combobox.currentIndex()}'
        )
        message_box.setStandardButtons(QMessageBox.StandardButton.Cancel)
        bt_primary = message_box.addButton('Primary', QMessageBox.ButtonRole.AcceptRole)
        bt_secondary = message_box.addButton(
            'Secondary', QMessageBox.ButtonRole.AcceptRole
        )
        message_box.exec_()
        palettes_primary = self.main_gui.get_tileset_palettes(True)
        palettes_secondary = self.main_gui.get_tileset_palettes(False)

        assert self.main_gui.project is not None
        palette = (
            render.pack_colors(palettes_primary)
            + render.pack_colors(palettes_secondary)
        )[self.tiles_palette_combobox.currentIndex()]

        if message_box.clickedButton() == bt_primary:
            primary = True
        elif message_box.clickedButton() == bt_secondary:
            primary = False
        else:
            return
        label = self.main_gui.get_tileset_gfx_label(primary)
        img = self.main_gui.project.load_gfx(primary, label)
        self.main_gui.project.save_gfx(primary, img, palette, label)

    def copy_behaviour(self):
        """Copies the behaviour properties of the current block."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.behaviour_clipboard = self.block_properties.model_value
        self.behaviour_clipboard_paste.setEnabled(True)

    def paste_behaviour(self):
        """Pastes the behaviour properties in the current clipboard."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        self.block_properties.set_value(self.behaviour_clipboard)

    def clear_behaviour(self):
        """Clears the behaviour properties of the current block."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        config = self.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]
        datatype = self.main_gui.project.model[config['behaviour_datatype']]
        tileset = (
            self.main_gui.tileset_primary
            if self.selected_block_idx < 0x280
            else self.main_gui.tileset_secondary
        )
        # Create a new "empty" instance
        empty = datatype(
            self.main_gui.project,
            config['behaviours_path'] + [self.selected_block_idx % 0x280],
            properties.get_parents_by_path(
                tileset, config['behaviours_path'] + [self.selected_block_idx % 0x280]
            ),
        )
        self.block_properties.set_value(empty)

    def import_palette(self):
        """Prompts a dialoge to select an image to import a palette from."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Palette from File',
            str(
                Path(
                    self.main_gui.settings.value('palette/recent', '.', str)  # type: ignore
                ).parent
                / 'palette.png'
            ),
            '4BPP PNG files (*.png)',
        )
        self.main_gui.settings.setValue('palette/recent', path)
        _, palette = agbimage.from_file(path)
        palette = palette.to_data()
        pal_idx = self.tiles_palette_combobox.currentIndex()
        palette_old = self.main_gui.get_tileset_palette(pal_idx)
        self.undo_stack.push(history.SetPalette(self, pal_idx, palette, palette_old))  # type: ignore
