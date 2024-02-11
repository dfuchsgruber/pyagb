from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import numpy as np
from PIL import ImageQt
from PIL import Image
from . import map_widget, properties, render, blocks, resource_tree, history
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from itertools import product
from agb import image as agbimage
import os

HFLIP = 1
VFLIP = 2

class TilesetWidget(QWidget):
    """ Class to model tilesets. """

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.undo_stack = QUndoStack()
        self.behaviour_clipboard = None
        layout = QGridLayout()
        self.setLayout(layout)

        blocks_group = QGroupBox('Blocks')
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QGraphicsView()
        self.blocks_scene_view.setViewport(QGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        blocks_layout = QGridLayout()
        blocks_group.setLayout(blocks_layout)
        blocks_layout.addWidget(self.blocks_scene_view)
        layout.addWidget(blocks_group, 2, 1, 4, 1)

        gfx_group = QGroupBox('Gfx')
        gfx_layout = QGridLayout()
        gfx_group.setLayout(gfx_layout)
        gfx_layout.addWidget(QLabel('Primary'), 1, 1, 1, 1)
        self.gfx_primary_combobox = QComboBox()
        gfx_layout.addWidget(self.gfx_primary_combobox, 1, 2, 1, 1)
        gfx_layout.addWidget(QLabel('Secondary'), 2, 1, 1, 1)
        self.gfx_secondary_combobox = QComboBox()
        gfx_layout.addWidget(self.gfx_secondary_combobox, 2, 2, 1, 1)
        self.gfx_primary_combobox.currentTextChanged.connect(lambda label: self.main_gui.change_gfx(label, True))
        self.gfx_secondary_combobox.currentTextChanged.connect(lambda label: self.main_gui.change_gfx(label, False))
        gfx_layout.setColumnStretch(1, 0)
        gfx_layout.setColumnStretch(2, 1)
        layout.addWidget(gfx_group, 2, 2, 1, 1)

        properties_group = QGroupBox('Properties')
        properties_layout = QGridLayout()
        properties_group.setLayout(properties_layout)
        self.properties_tree_tsp = TilesetProperties(self, True)
        properties_layout.addWidget(self.properties_tree_tsp, 0, 0, 1, 1)
        self.properties_tree_tss = TilesetProperties(self, False)
        properties_layout.addWidget(self.properties_tree_tss, 0, 1, 1, 1)
        layout.addWidget(properties_group, 3, 2, 1, 1)

        seleciton_group = QGroupBox('Selection')
        selection_layout = QGridLayout()
        seleciton_group.setLayout(selection_layout)
        self.selection_scene = SelectionScene(self)
        self.selection_scene_view = QGraphicsView()
        self.selection_scene_view.setViewport(QGLWidget())
        self.selection_scene_view.setScene(self.selection_scene)
        selection_layout.addWidget(self.selection_scene_view)
        layout.addWidget(seleciton_group, 4, 2, 1, 1)

        current_block_group = QGroupBox('Current Block')
        current_block_layout = QGridLayout()
        current_block_group.setLayout(current_block_layout)
        lower_group = QGroupBox('Bottom Layer')
        lower_layout = QGridLayout()
        lower_group.setLayout(lower_layout)
        self.block_lower_scene = BlockScene(self, 0)
        self.block_lower_scene_view = QGraphicsView()
        self.block_lower_scene_view.setViewport(QGLWidget())
        self.block_lower_scene_view.setScene(self.block_lower_scene)
        lower_layout.addWidget(self.block_lower_scene_view)
        mid_group = QGroupBox('Mid Layer')
        mid_layout = QGridLayout()
        mid_group.setLayout(mid_layout)
        self.block_mid_scene = BlockScene(self, 1)
        self.block_mid_scene_view = QGraphicsView()
        self.block_mid_scene_view.setViewport(QGLWidget())
        self.block_mid_scene_view.setScene(self.block_mid_scene)
        mid_layout.addWidget(self.block_mid_scene_view)
        upper_group = QGroupBox('Upper Layer')
        upper_layout = QGridLayout()
        upper_group.setLayout(upper_layout)
        self.block_upper_scene = BlockScene(self, 2)
        self.block_upper_scene_view = QGraphicsView()
        self.block_upper_scene_view.setViewport(QGLWidget())
        self.block_upper_scene_view.setScene(self.block_upper_scene)
        upper_layout.addWidget(self.block_upper_scene_view)
        current_block_layout.addWidget(lower_group, 1, 1, 1, 1)
        current_block_layout.addWidget(mid_group, 1, 2, 1, 1)
        current_block_layout.addWidget(upper_group, 1, 3, 1, 1)
        self.block_properties = BlockProperties(self)
        current_block_layout.addWidget(self.block_properties, 2, 1, 1, 3)
        behaviour_clipboard_layout = QGridLayout()
        current_block_layout.addLayout(behaviour_clipboard_layout, 3, 1, 1, 2)
        self.behaviour_clipboard_copy = QPushButton('Copy')
        self.behaviour_clipboard_copy.clicked.connect(self.copy_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_copy, 1, 1)
        self.behaviour_clipboard_paste = QPushButton('Paste')
        self.behaviour_clipboard_paste.clicked.connect(self.paste_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_paste, 1, 2)
        self.behaviour_clipboard_clear = QPushButton('Clear')
        self.behaviour_clipboard_clear.clicked.connect(self.clear_behaviour)
        behaviour_clipboard_layout.addWidget(self.behaviour_clipboard_clear, 1, 3)
        behaviour_clipboard_layout.addWidget(QLabel(), 1, 4)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        layout.setColumnStretch(4, 1)

        current_block_layout.setRowStretch(1, 0)
        current_block_layout.setRowStretch(2, 1)
        current_block_layout.setRowStretch(3, 0)

        layout.addWidget(current_block_group, 5, 2, 1, 1)

        tiles_group = QGroupBox('Tiles')
        tiles_layout = QGridLayout()
        tiles_group.setLayout(tiles_layout)

        self.tiles_mirror_horizontal_checkbox = QCheckBox('H-Flip')
        tiles_layout.addWidget(self.tiles_mirror_horizontal_checkbox, 1, 2, 1, 1)
        self.tiles_mirror_horizontal_checkbox.toggled.connect(self.update_tiles)
        self.tiles_mirror_vertical_checkbox = QCheckBox('V-Flip')
        tiles_layout.addWidget(self.tiles_mirror_vertical_checkbox, 1, 3, 1, 1)
        self.tiles_mirror_vertical_checkbox.toggled.connect(self.update_tiles)
        tiles_palette_group = QGroupBox('Palette')
        tiles_palette_group_layout = QGridLayout()
        tiles_palette_group.setLayout(tiles_palette_group_layout)
        self.tiles_palette_combobox = QComboBox()
        self.tiles_palette_combobox.addItems(map(str, range(13)))
        self.tiles_palette_combobox.currentIndexChanged.connect(self.update_tiles)
        tiles_palette_group_layout.addWidget(self.tiles_palette_combobox, 1, 1, 1, 1)
        tiles_import_button = QPushButton('Import')
        tiles_import_button.clicked.connect(self.import_palette)
        tiles_palette_group_layout.addWidget(tiles_import_button, 1, 2, 1, 1)
        tiles_export_button = QPushButton('Export')
        tiles_export_button.clicked.connect(self.export_gfx)
        tiles_palette_group_layout.addWidget(tiles_export_button, 1, 3, 1, 1)
        tiles_palette_group_layout.setColumnStretch(1, 0)
        tiles_palette_group_layout.setColumnStretch(2, 0)
        tiles_palette_group_layout.setColumnStretch(3, 0)

        tiles_layout.addWidget(tiles_palette_group, 1, 1, 1, 1)
        self.tiles_scene = TilesScene(self)
        self.tiles_scene_view = QGraphicsView()
        self.tiles_scene_view.setViewport(QGLWidget())
        self.tiles_scene_view.setScene(self.tiles_scene)
        tiles_layout.addWidget(self.tiles_scene_view, 3, 1, 1, 3)
        tiles_layout.setColumnStretch(1, 1)
        tiles_layout.setColumnStretch(2, 0)
        tiles_layout.setColumnStretch(3, 0)
        layout.addWidget(tiles_group, 2, 3, 4, 1)

        zoom_group = QGroupBox('Zoom')
        zoom_layout = QGridLayout()
        zoom_group.setLayout(zoom_layout)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(5)
        self.zoom_slider.setMaximum(40)
        self.zoom_slider.setTickInterval(1)
        zoom_layout.addWidget(self.zoom_slider, 1, 1, 1, 1)
        self.zoom_label = QLabel()
        zoom_layout.addWidget(self.zoom_label, 1, 2, 1, 1)
        layout.addWidget(zoom_group, 1, 1, 1, 3)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        self.zoom_slider.setValue(self.main_gui.settings['tileset.zoom'])

        self.info_label = QLabel('')
        layout.addWidget(self.info_label, 6, 1, 1, 3)

        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 1)
        layout.setRowStretch(4, 0)
        layout.setRowStretch(5, 3)
        layout.setRowStretch(6, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        self.load_project() # Initialize widgets as disabled

    def load_project(self):
        """ Loads a new project. """
        if self.main_gui.project is None: return
        self.gfx_primary_combobox.blockSignals(True)
        self.gfx_primary_combobox.clear()
        self.gfx_primary_combobox.addItems(list(self.main_gui.project.gfxs_primary.keys()))
        self.gfx_primary_combobox.blockSignals(False)
        self.gfx_secondary_combobox.blockSignals(True)
        self.gfx_secondary_combobox.clear()
        self.gfx_secondary_combobox.addItems(list(self.main_gui.project.gfxs_secondary.keys()))
        self.gfx_secondary_combobox.blockSignals(False)
        self.load_header()
        
    def load_header(self):
        """ Updates the blocks of a new header. """
        self.tiles_scene.clear()
        self.blocks_scene.clear()
        self.selection_scene.clear()
        self.blocks_scene.clear()
        self.selected_block = 0
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: 
            self.behaviour_clipboard_paste.setEnabled(False)
            self.behaviour_clipboard_copy.setEnabled(False)
            self.behaviour_clipboard_clear.setEnabled(False)
            self.gfx_primary_combobox.setEnabled(False)
            self.gfx_secondary_combobox.setEnabled(False)
            # self.animation_primary_line_edit.setEnabled(False)
            # self.animation_secondary_line_edit.setEnabled(False)
            self.properties_tree_tsp.setEnabled(False)
            self.properties_tree_tss.setEnabled(False)
        else:
            self.behaviour_clipboard_copy.setEnabled(True)
            self.behaviour_clipboard_clear.setEnabled(True)
            self.gfx_primary_combobox.setEnabled(True)
            self.gfx_secondary_combobox.setEnabled(True)
            gfx_primary_label = properties.get_member_by_path(self.main_gui.tileset_primary, self.main_gui.project.config['pymap']['tileset_primary']['gfx_path'])
            gfx_secondary_label = properties.get_member_by_path(self.main_gui.tileset_secondary, self.main_gui.project.config['pymap']['tileset_secondary']['gfx_path'])
            self.gfx_primary_combobox.blockSignals(True)
            self.gfx_primary_combobox.setCurrentText(gfx_primary_label)
            self.gfx_primary_combobox.blockSignals(False)
            self.gfx_secondary_combobox.blockSignals(True)
            self.gfx_secondary_combobox.setCurrentText(gfx_secondary_label)
            self.gfx_secondary_combobox.blockSignals(False)
            
            self.properties_tree_tsp.setEnabled(True)
            self.properties_tree_tss.setEnabled(True)
            self.properties_tree_tsp.load_tileset()
            self.properties_tree_tss.load_tileset()
            
            self.load_tiles()
            self.load_blocks()
            self.set_selection(np.array([[
                self._empty_block_tile()
            ]]))
            self.set_current_block(0)
    
    def _empty_block_tile(self):
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        config = self.main_gui.project.config['pymap']['tileset_primary' if self.selected_block < 0x280 else 'tileset_secondary']
        datatype = self.main_gui.project.model[config['block_datatype']]
        tileset = self.main_gui.tileset_primary if self.selected_block < 0x280 else self.main_gui.tileset_secondary
        # Create a new "empty" instance
        return datatype(
            self.main_gui.project, 
            config['blocks_path'] + [self.selected_block % 0x280, 0], 
            properties.get_parents_by_path(tileset, config['blocks_path'] + [self.selected_block % 0x280, 0]),
            )

    def reload(self):
        """ Reloads the entire view (in case tiles or gfx have changed). """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.load_tiles()
        self.load_blocks()
        self.set_current_block(self.selected_block)
        self.set_selection(self.selection)

    def load_tiles(self):
        """ Reloads the tiles. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.tile_pixmaps = []
        for palette_idx in range(13):
            pixmaps = {}
            for flip in range(4):
                # Assemble the entire picture
                image = Image.new('RGBA', (128, 512))
                for idx, tile_img in enumerate(self.main_gui.tiles[palette_idx]):
                    x, y = idx % 16, idx // 16
                    if flip & HFLIP:
                        tile_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
                        x = 15 - x
                    if flip & VFLIP:
                        tile_img = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
                        y = 63 - y
                    image.paste(tile_img, box=(8 * x, 8 * y))
                pixmaps[flip] = QPixmap.fromImage(ImageQt(image.convert('RGB').convert('RGBA')))
            self.tile_pixmaps.append(pixmaps)
        self.update_tiles()

    def load_blocks(self):
        """ Reloads the blocks. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.blocks_image = render.draw_blocks(self.main_gui.blocks)
        self.update_blocks()
    
    def set_current_block(self, block_idx):
        """ Reloads the current block. """
        self.block_lower_scene.clear()
        self.block_mid_scene.clear()
        self.block_upper_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.selected_block = block_idx
        self.block_lower_scene.update_block()
        self.block_mid_scene.update_block()
        self.block_upper_scene.update_block()
        self.blocks_scene.update_selection_rect()
        self.block_properties.load_block()

    def update_blocks(self):
        """ Updates the display of the blocks. """
        self.blocks_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        width, height = self.blocks_image.size
        width, height = int(width * self.zoom_slider.value() / 10), int(height * self.zoom_slider.value() / 10)
        pixmap = QPixmap.fromImage(ImageQt(self.blocks_image)).scaled(width, height)
        item = QGraphicsPixmapItem(pixmap)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        # Add the selection rectangle
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        self.blocks_scene.selection_rect = self.blocks_scene.addRect(0, 0, 0, 0, pen = QPen(color, 1.0 * self.zoom_slider.value() / 10), brush = QBrush(0))
        self.blocks_scene.update_selection_rect()

    def update_tiles(self):
        """ Updates the display of the tiles widget. """
        self.tiles_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        width, height = int(128 * self.zoom_slider.value() / 10), int(512 * self.zoom_slider.value() / 10)
        flip = (HFLIP if self.tiles_mirror_horizontal_checkbox.isChecked() else 0) | (VFLIP if self.tiles_mirror_vertical_checkbox.isChecked() else 0)
        item = QGraphicsPixmapItem(self.tile_pixmaps[self.tiles_palette_combobox.currentIndex()][flip].scaled(width, height))
        self.tiles_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        # Add the selection rectangle
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        self.tiles_scene.selection_rect = self.tiles_scene.addRect(0, 0, 0, 0, pen = QPen(color, 1.0 * self.zoom_slider.value() / 10), brush = QBrush(0))
        self.tiles_scene.update_selection_rect()
        self.tiles_scene.setSceneRect(0, 0, width, height)
        
    def zoom_changed(self):
        """ Event handler for when the zoom has changed. """
        self.zoom_label.setText(f'{self.zoom_slider.value() * 10}%')
        self.main_gui.settings['tileset.zoom'] = self.zoom_slider.value()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.update_tiles()
        self.update_blocks()
        self.set_selection(self.selection)
        self.block_lower_scene.update_block()
        self.block_mid_scene.update_block()
        self.block_upper_scene.update_block()

    def set_selection(self, selection):
        """ Sets the selection to a set of tiles. """
        self.selection = selection
        self.selection_scene.clear()
        image = Image.new('RGBA', (8 * selection.shape[1], 8 * selection.shape[0]))
        for (y, x), tile in np.ndenumerate(selection):
            tile_img = self.main_gui.tiles[tile['palette_idx']][tile['tile_idx']]
            if tile['horizontal_flip']:
                tile_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
            if tile['vertical_flip']:
                tile_img = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
            image.paste(tile_img, box=(8 * x, 8 * y))
        width, height = int(8 * selection.shape[1] * self.zoom_slider.value() / 10), int(8 * selection.shape[0] * self.zoom_slider.value() / 10)
        item = QGraphicsPixmapItem(QPixmap.fromImage(ImageQt(image.convert('RGB').convert('RGBA'))).scaled(width, height))
        self.selection_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        self.selection_scene.setSceneRect(0, 0, width, height)
            
    def set_info(self, tile_idx):
        """ Updates the info to a specific tile index or clears it if None is given. """
        if tile_idx is None:
            self.info_label.setText('')
        else:
            section = 'Primary Tileset' if tile_idx < 640 else ('Secondary Tileset' if tile_idx < 1020 else 'Reserved for Doors')
            self.info_label.setText(f'Tile {hex(tile_idx)}, {section}')

    def export_gfx(self):
        """ Prompts the user to export to export a gfx in the selected palette. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        message_box = QMessageBox(self)
        message_box.setWindowTitle(f'Export Gfx in Palette {self.tiles_palette_combobox.currentIndex()}')
        message_box.setText(f'Select which gfx to export in palette {self.tiles_palette_combobox.currentIndex()}')
        message_box.setStandardButtons(QMessageBox.Cancel)
        bt_primary = message_box.addButton('Primary', QMessageBox.AcceptRole)
        bt_secondary = message_box.addButton('Secondary', QMessageBox.AcceptRole)
        message_box.exec_()

        # Fetch current palettes
        palettes_primary = properties.get_member_by_path(self.main_gui.tileset_primary, self.main_gui.project.config['pymap']['tileset_primary']['palettes_path'])
        palettes_secondary = properties.get_member_by_path(self.main_gui.tileset_secondary, self.main_gui.project.config['pymap']['tileset_secondary']['palettes_path'])
        palette = (render.pack_colors(palettes_primary, self.main_gui.project) + render.pack_colors(palettes_secondary, self.main_gui.project))[self.tiles_palette_combobox.currentIndex()]

        if message_box.clickedButton() == bt_primary:
            label = properties.get_member_by_path(self.main_gui.tileset_primary, self.main_gui.project.config['pymap']['tileset_primary']['gfx_path'])
            img = self.main_gui.project.load_gfx(True, label)
            self.main_gui.project.save_gfx(True, img, palette, label)
        elif message_box.clickedButton() == bt_secondary:
            label = properties.get_member_by_path(self.main_gui.tileset_secondary, self.main_gui.project.config['pymap']['tileset_secondary']['gfx_path'])
            img = self.main_gui.project.load_gfx(False, label)
            self.main_gui.project.save_gfx(False, img, palette, label)

    def copy_behaviour(self):
        """ Copies the behaviour properties of the current block. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.behaviour_clipboard = self.block_properties.get_value()
        self.behaviour_clipboard_paste.setEnabled(True)

    def paste_behaviour(self):
        """ Pastes the behaviour properties in the current clipboard. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.block_properties.set_value(self.behaviour_clipboard)

    def clear_behaviour(self):
        """ Clears the behaviour properties of the current block. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        config = self.main_gui.project.config['pymap']['tileset_primary' if self.selected_block < 0x280 else 'tileset_secondary']
        datatype = self.main_gui.project.model[config['behaviour_datatype']]
        tileset = self.main_gui.tileset_primary if self.selected_block < 0x280 else self.main_gui.tileset_secondary
        # Create a new "empty" instance
        empty = datatype(
            self.main_gui.project, 
            config['behaviours_path'] + [self.selected_block % 0x280], 
            properties.get_parents_by_path(tileset, config['behaviours_path'] + [self.selected_block % 0x280]),
            )
        self.block_properties.set_value(empty)

    def import_palette(self):
        """ Prompts a dialoge that asks the user to select a 4bpp png to import a palette from. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        path, suffix = QFileDialog.getOpenFileName(
            self, 'Import Palette from File', os.path.join(os.path.dirname(self.main_gui.settings['recent.palette']), 
            f'palette.png'), '4BPP PNG files (*.png)')
        self.main_gui.settings['recent.palette'] = path
        _, palette = agbimage.from_file(path)
        palette = palette.to_data()
        pal_idx = self.tiles_palette_combobox.currentIndex()
        if pal_idx < 7:
            palette_old = properties.get_member_by_path(self.main_gui.tileset_primary, self.main_gui.project.config['pymap']['tileset_primary']['palettes_path'])[pal_idx]
        else:
            palette_old = properties.get_member_by_path(self.main_gui.tileset_secondary, self.main_gui.project.config['pymap']['tileset_secondary']['palettes_path'])[pal_idx % 7]
        self.undo_stack.push(history.SetPalette(self, pal_idx, palette, palette_old))

class SelectionScene(QGraphicsScene):
    """ Scene for the selected tiles. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value())
        height, width = self.tileset_widget.selection.shape
        if width > x >= 0 and height > y >= 0:
            self.tileset_widget.set_info(self.tileset_widget.selection[y, x]['tile_idx'])
        else:
            self.tileset_widget.set_info(None)


class BlocksScene(QGraphicsScene):
    """ Scene for the individual blocks. """

    def __init__(self, tileset_widget: 'TilesetWidget', parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.selection_rect = None
        self.clipboard = None

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        if 8 > x >= 0 and 128 > y >= 0:
            self.tileset_widget.info_label.setText(f'Block {hex(8 * y + x)}')
        else:
            self.tileset_widget.info_label.setText('')

    def mousePressEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        if 8 > x >= 0 and 128 > y >= 0 and (event.button() == Qt.RightButton or event.button() == Qt.LeftButton):
            self.tileset_widget.set_current_block(8 * y + x)

    def update_selection_rect(self):
        """ Updates the selection rectangle. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None  or self.selection_rect is None: return
        x, y = self.tileset_widget.selected_block % 8, self.tileset_widget.selected_block // 8
        size = 16 * self.tileset_widget.zoom_slider.value() / 10
        x, y = int(x * size), int(y * size)
        self.selection_rect.setRect(x, y, int(size), int(size))
        self.setSceneRect(0, 0, int(8 * size), int(128 * size))

    def contextMenuEvent(self, event: 'QGraphicsSceneContextMenuEvent') -> None:
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        block_idx = 8 * y + x
        block = np.array(properties.get_member_by_path(
            self.tileset_widget.main_gui.tileset_primary if block_idx < 0x280 else self.tileset_widget.main_gui.tileset_secondary, 
            self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if block_idx < 0x280 else 'tileset_secondary']['blocks_path']
            )[block_idx % 0x280]).reshape(3, 2, 2).copy()
        def paste(paste_tiles, paste_behaviour):
            """ Helper to paste either the block tiles and / or the block behaviour """
            block_clipboard, behaviour_clipboard = self.clipboard
            self.tileset_widget.undo_stack.beginMacro('Paste Block')
            if paste_behaviour:
                self.tileset_widget.block_properties.set_value(behaviour_clipboard)
            if paste_tiles:
                for layer in range(3):
                    self.tileset_widget.undo_stack.push(history.SetTiles(
                        self.tileset_widget, block_idx, layer, 0, 0, block_clipboard[layer].copy(), block[layer].copy()
                    ))
            self.tileset_widget.undo_stack.endMacro()

        def clear(clear_tiles, clear_behaviour):
            """ Helper to clear either the block tiles / or the block behaviour """
            self.tileset_widget.undo_stack.beginMacro('Clear Block')
            if clear_behaviour:
                self.tileset_widget.clear_behaviour()
            if clear_tiles:
                for layer in range(3):
                    self.tileset_widget.undo_stack.push(history.SetTiles(
                        self.tileset_widget, block_idx, layer, 0, 0, np.array([self.tileset_widget._empty_block_tile() for _ in range(4)]).reshape((2, 2)), block[layer].copy()
                    ))
            self.tileset_widget.undo_stack.endMacro()

        menu = QMenu()
        copy_action = menu.addAction('Copy')
        menu.addSeparator()
        paste_action = menu.addAction('Paste')
        paste_tiles_action = menu.addAction('Paste Tiles')
        # Behaviour pasting / clearing should not be done here, right? Use the tree widget instead...
        # paste_behaviour_action = menu.addAction('Paste Behaviour')
        menu.addSeparator()
        clear_all_action = menu.addAction('Clear')
        clear_tiles_action = menu.addAction('Clear Tiles')
        if self.clipboard is None:
            paste_action.setEnabled(False)
            paste_tiles_action.setEnabled(False)
        action = menu.exec(event.screenPos())
        if action == copy_action:
            # Copy both the block data and its associated behaviour
            self.clipboard = block, self.tileset_widget.block_properties.get_value()
        elif action == paste_action:
            paste(True, True)
        elif action == paste_tiles_action:
            paste(True, False)
        elif action == clear_all_action:
            clear(True, True)
        elif action == clear_tiles_action:
            clear(True, False)
        

class TilesScene(QGraphicsScene):
    """ Scene for the individual tiles. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.selection_box = None
        self.selection_rect = None

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value())
        if 16 > x >= 0 and 64 > y >= 0:
            self.tileset_widget.set_info(16 * y + x)
            if self.selection_box is not None:
                x0, x1, y0, y1 = self.selection_box
                if x1 != x + 1 or y1 != y + 1:
                    # Redraw the selection
                    self.selection_box = x0, x + 1, y0, y + 1
                    self.select_tiles()
        else:
            self.tileset_widget.set_info(None)
        
    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value())
        if 16 > x >= 0 and 64 > y >= 0:
            if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
                # Select this tile as starting point
                self.selection_box = x, x + 1, y, y + 1
                self.select_tiles()

    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.RightButton or event.button() == Qt.LeftButton:
            self.selection_box = None

    def select_tiles(self):
        """ Updates the selection according to the current selection box. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        flip = (int(self.tileset_widget.tiles_mirror_horizontal_checkbox.isChecked()) * HFLIP) | (int(self.tileset_widget.tiles_mirror_vertical_checkbox.isChecked()) * VFLIP)
        if self.selection_box is not None:
            self.tileset_widget.set_selection(render.select_blocks(
                tiles[self.tileset_widget.tiles_palette_combobox.currentIndex(), flip], *self.selection_box
            ))
        self.update_selection_rect()
    
    def update_selection_rect(self):
        """ Updates the selection rectangle. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None  or self.selection_rect is None: return
        if self.selection_box is not None:
            # Redraw the red selection box
            x0, x1, y0, y1 = render.get_box(*self.selection_box)
            scale = 8 * self.tileset_widget.zoom_slider.value() / 10
            self.selection_rect.setRect(int(scale * x0), int(scale * y0), int(scale * (x1 - x0)), int(scale * (y1 - y0)))
        else:
            self.selection_rect.setRect(0, 0, 0, 0)

# A static num_pals x num_flips x 64 x 16 array that represents the display of the tileset widget
tiles = np.array([
    [
        [
            {
                'tile_idx' : tile_idx,
                'palette_idx' : pal_idx,
                'horizontal_flip' : int((flip & HFLIP) > 0),
                'vertical_flip' : int((flip & VFLIP) > 0),
            }
            for tile_idx in range(0x400)
        ]
        for flip in range(4)
    ]
    for pal_idx in range(13)
]).reshape((13, 4, 64, 16))
for flip in range(4):
    if flip & HFLIP:
        tiles[:, flip, :, :] = tiles[:, flip, :, ::-1]
    if flip & VFLIP:
        tiles[:, flip, :, :] = tiles[:, flip, ::-1, :]



class BlockScene(QGraphicsScene):
    """ Scene for the current block. """

    def __init__(self, tileset_widget: TilesetWidget, layer: int, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.layer = layer
        self.last_draw = None
        self.selection_box = None

    def update_block(self):
        """ Updates the display of this block"""
        self.clear()
        block = np.array(properties.get_member_by_path(
            self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary, 
            self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']['blocks_path']
            )[self.tileset_widget.selected_block % 0x280])
        block = block[self.layer * 4 : (self.layer + 1) * 4]
        image = Image.new('RGBA', (16, 16))
        for (y, x), tile in np.ndenumerate(block.reshape(2, 2)):
            tile_img = self.tileset_widget.main_gui.tiles[tile['palette_idx']][tile['tile_idx']]
            if tile['horizontal_flip']:
                tile_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
            if tile['vertical_flip']:
                tile_img = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
            image.paste(tile_img, box=(8 * x, 8 * y))
        size = int(self.tileset_widget.zoom_slider.value() * 16 / 10)
        item = QGraphicsPixmapItem(QPixmap.fromImage(ImageQt(image.convert('RGB').convert('RGBA')).scaled(size, size)))
        self.addItem(item)
        item.setAcceptHoverEvents(True)
        self.setSceneRect(0, 0, size, size)

    def update_selection_box(self):
        """ Pastes the selection box to the current selection. """
        block = np.array(properties.get_member_by_path(
            self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary, 
            self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']['blocks_path']
            )[self.tileset_widget.selected_block % 0x280]).reshape(3, 2, 2)
        self.tileset_widget.set_selection(render.select_blocks(
            block[int(self.layer)], *self.selection_box))
        
    def mouseMoveEvent(self, event):
        """ Event handler for hover events on the map image. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value())
        if 2 > x >= 0 and 2 > y >= 0:
            if self.last_draw is not None and self.last_draw != (x, y):
                self.last_draw = x, y
                selection_height, selection_width = self.tileset_widget.selection.shape
                # Trim the selection to fit into the 2 x 2 window
                selection = self.tileset_widget.selection[: min(2 - y, selection_height), : min(2 - x, selection_width)].copy()
                block = np.array(properties.get_member_by_path(
                    self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary, 
                    self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']['blocks_path']
                    )[self.tileset_widget.selected_block % 0x280]).reshape(3, 2, 2)
                # Extract the old tiles
                tiles_old = block[int(self.layer), y : y + selection.shape[0], x : x + selection.shape[1]].copy()
                self.tileset_widget.undo_stack.push(history.SetTiles(
                    self.tileset_widget, self.tileset_widget.selected_block, int(self.layer), x, y, selection.copy(), tiles_old.copy()
                ))
            if self.selection_box is not None:
                x0, x1, y0, y1 = self.selection_box
                if x1 != x + 1 or y1 != y + 1:
                    self.selection_box = x0, x + 1, y0, y + 1
                    self.update_selection_box()
                    # Clear the selection box of the tiles widget
                    self.tileset_widget.tiles_scene.selection_box = None
                    self.tileset_widget.tiles_scene.select_tiles()


    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()), int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value())
        if 2 > x >= 0 and 2 > y >= 0:
            if event.button() == Qt.LeftButton:
                self.last_draw = -1, -1 # This triggers the drawing routine
                self.tileset_widget.undo_stack.beginMacro('Drawing Tiles')
                self.mouseMoveEvent(event)
            elif event.button() == Qt.RightButton:
                self.selection_box = x, x + 1, y, y + 1
                self.update_selection_box()
                # Select the palette of this tile
                pal_idx = self.tileset_widget.selection[0, 0]['palette_idx']
                self.tileset_widget.tiles_palette_combobox.setCurrentIndex(pal_idx)
                # Select the tile in the tiles widget
                tile_idx = self.tileset_widget.selection[0, 0]['tile_idx']
                x, y = tile_idx % 16, tile_idx // 16
                hflip, vflip = self.tileset_widget.tiles_mirror_horizontal_checkbox.isChecked(), self.tileset_widget.tiles_mirror_vertical_checkbox.isChecked()
                if hflip:
                    x = 15 - x
                if vflip:
                    y = 63 - y
                self.tileset_widget.tiles_scene.selection_box = x, x + 1, y, y + 1
                self.tileset_widget.tiles_scene.select_tiles()
                self.tileset_widget.tiles_scene.selection_box = None
                # Ensure the rect is visible
                scale = int(self.tileset_widget.zoom_slider.value() * 8 / 10)
                self.tileset_widget.tiles_scene_view.ensureVisible(self.tileset_widget.tiles_scene.selection_rect.rect())


    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None: return
        if event.button() == Qt.LeftButton:
            self.last_draw = None
            self.tileset_widget.undo_stack.endMacro()
        elif event.button() == Qt.RightButton:
            self.selection_box = None

class TilesetProperties(ParameterTree):
    """ Tree to display additional properties of the tilesets """

    def __init__(self, tileset_widget, is_primary, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary
        self.header().setSectionResizeMode(QHeaderView.Interactive)
        self.header().setStretchLastSection(True)
        self.root = None

    def load_tileset(self):
        self.clear()
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None:
            self.root = None
        else:
            config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.is_primary else 'tileset_secondary']
            datatype = config['datatype']
            tileset = self.tileset_widget.main_gui.tileset_primary if self.is_primary else self.tileset_widget.main_gui.tileset_secondary
            self.root = properties.type_to_parameter(self.tileset_widget.main_gui.project, datatype)(
                '.', self.tileset_widget.main_gui.project, datatype, tileset, [], [],
            )
            self.addParameters(self.root, showTop=False)
            self.root.sigTreeStateChanged.connect(self.tree_changed)

    def update(self):
        """ Updates all values in the tree according to the current properties. """
        # config = self.tileset_widget.main_gui.self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.is_primary else 'tileset_secondary']
        tileset = self.tileset_widget.main_gui.tileset_primary if self.is_primary else self.tileset_widget.main_gui.tileset_secondary
        self.root.blockSignals(True)
        self.root.update(tileset)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        root = self.tileset_widget.main_gui.tileset_primary if self.is_primary else self.tileset_widget.main_gui.tileset_secondary
        diffs = DeepDiff(root, self.root.model_value())
        statements_redo = []
        statements_undo = []
        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:
                    value_new = diffs[change][path]['new_value']
                    value_old = diffs[change][path]['old_value']
                    statements_redo.append(f'{path} = \'{value_new}\'')
                    statements_undo.append(f'{path} = \'{value_old}\'')
                    self.tileset_widget.undo_stack.push(history.ChangeTilesetProperty(
                        self.tileset_widget, self.is_primary, statements_redo, statements_undo
                    ))

    def get_value(self):
        """ Gets the model value of the current block or None if no block is selected. """
        if self.root is None: return None
        return self.root.model_value()

    def set_value(self, behaviour):
        """ Replaces the entry properties of the current block if one is selected. """
        if self.model is None: return
        self.root.blockSignals(True)
        self.root.update(behaviour)
        self.root.blockSignals(False)
        self.tree_changed(None)

class BlockProperties(ParameterTree):
    """ Tree to display block properties. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.setHeaderLabels(['Property', 'Value'])
        self.header().setSectionResizeMode(QHeaderView.Interactive)
        self.header().setStretchLastSection(True)
        self.root = None

    def load_block(self):
        """ Loads the currently displayed blocks properties. """
        self.clear()
        if self.tileset_widget.main_gui.project is None or self.tileset_widget.main_gui.header is None or self.tileset_widget.main_gui.footer is None or \
            self.tileset_widget.main_gui.tileset_primary is None or self.tileset_widget.main_gui.tileset_secondary is None:
            self.root = None
        else:
            config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']
            tileset = self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary
            datatype = config['behaviour_datatype']
            behaviours = properties.get_member_by_path(tileset, config['behaviours_path'])
            behaviour = behaviours[self.tileset_widget.selected_block % 0x280]
            parents = properties.get_parents_by_path(tileset, config['behaviours_path'] + [self.tileset_widget.selected_block % 0x280])

            self.root = properties.type_to_parameter(self.tileset_widget.main_gui.project, datatype)(
                '.', self.tileset_widget.main_gui.project, datatype, behaviour, 
                config['behaviours_path'] + [self.tileset_widget.selected_block % 0x280], 
                parents)
            self.addParameters(self.root, showTop=False)
            self.root.sigTreeStateChanged.connect(self.tree_changed)
        
    def update(self):
        """ Updates all values in the tree according to the current properties. """
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']
        tileset = self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary
        behaviour = properties.get_member_by_path(tileset, config['behaviours_path'])[self.tileset_widget.selected_block % 0x280]
        self.root.blockSignals(True)
        self.root.update(behaviour)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.tileset_widget.selected_block < 0x280 else 'tileset_secondary']
        tileset = self.tileset_widget.main_gui.tileset_primary if self.tileset_widget.selected_block < 0x280 else self.tileset_widget.main_gui.tileset_secondary
        root = properties.get_member_by_path(tileset, config['behaviours_path'])[self.tileset_widget.selected_block % 0x280]
        diffs = DeepDiff(root, self.root.model_value())
        statements_redo = []
        statements_undo = []
        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:
                    value_new = diffs[change][path]['new_value']
                    value_old = diffs[change][path]['old_value']
                    statements_redo.append(f'{path} = \'{value_new}\'')
                    statements_undo.append(f'{path} = \'{value_old}\'')
                    self.tileset_widget.undo_stack.push(history.ChangeBlockProperty(
                        self.tileset_widget, self.tileset_widget.selected_block, statements_redo, statements_undo
                    ))

    def get_value(self):
        """ Gets the model value of the current block or None if no block is selected. """
        if self.root is None: return None
        return self.root.model_value()

    def set_value(self, behaviour):
        """ Replaces the entrie properties of the current block if one is selected. """
        if self.model is None: return
        self.root.blockSignals(True)
        self.root.update(behaviour)
        self.root.blockSignals(False)
        self.tree_changed(None)
        
class QLineEditWithoutHistory(QLineEdit):
    """ Subclass this thing in order to manually filter out undo events. """

    def event(self, event):
        if event.type() == QEvent.KeyPress:
            if event.matches(QKeySequence.Undo) or event.matches(QKeySequence.Redo):
                return False
        return super().event(event)