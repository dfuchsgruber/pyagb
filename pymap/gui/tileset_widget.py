from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import numpy as np
from PIL.ImageQt import ImageQt
from PIL import Image
import map_widget, properties, render, blocks, resource_tree, history
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from itertools import product

HFLIP = 1
VFLIP = 2

class TilesetWidget(QWidget):
    """ Class to model tilesets. """

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.undo_stack = QUndoStack()
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
        layout.addWidget(blocks_group, 2, 1, 3, 1)

        gfx_group = QGroupBox('Gfx')
        gfx_layout = QGridLayout()
        gfx_group.setLayout(gfx_layout)
        gfx_layout.addWidget(QLabel('Primary'), 1, 1, 1, 1)
        self.gfx_primary_combobox = QComboBox()
        gfx_layout.addWidget(self.gfx_primary_combobox, 1, 2, 1, 1)
        gfx_layout.addWidget(QLabel('Secondary'), 1, 3, 1, 1)
        self.gfx_secondary_combobox = QComboBox()
        gfx_layout.addWidget(self.gfx_secondary_combobox, 1, 4, 1, 1)
        gfx_layout.setColumnStretch(1, 0)
        gfx_layout.setColumnStretch(2, 1)
        gfx_layout.setColumnStretch(3, 0)
        gfx_layout.setColumnStretch(4, 1)
        layout.addWidget(gfx_group, 2, 2, 1, 1)

        seleciton_group = QGroupBox('Selection')
        selection_layout = QGridLayout()
        seleciton_group.setLayout(selection_layout)
        self.selection_scene = SelectionScene(self)
        self.selection_scene_view = QGraphicsView()
        self.selection_scene_view.setViewport(QGLWidget())
        self.selection_scene_view.setScene(self.selection_scene)
        selection_layout.addWidget(self.selection_scene_view)
        layout.addWidget(seleciton_group, 3, 2, 1, 1)

        current_block_group = QGroupBox('Current Block')
        current_block_layout = QGridLayout()
        current_block_group.setLayout(current_block_layout)
        layout.addWidget(current_block_group, 4, 2, 1, 1)

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
        tiles_palette_group_layout.addWidget(tiles_import_button, 1, 2, 1, 1)
        tiles_export_button = QPushButton('Export')
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
        layout.addWidget(tiles_group, 2, 3, 3, 1)

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
        layout.addWidget(self.info_label, 5, 1, 1, 3)

        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 0)
        layout.setRowStretch(4, 1)
        layout.setRowStretch(5, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

    def load_project(self):
        """ Loads a new project. """
        self.load_header()

    def load_header(self):
        """ Updates the blocks of a new header. """
        self.tiles_scene.clear()
        self.blocks_scene.clear()
        self.selection_scene.clear()
        self.blocks_scene.clear()
        # TODO: properties ?
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.load_tiles()
        self.load_blocks()
        
    def load_tiles(self):
        """ Reloads the tiles. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.tile_pixmaps = []
        for palette_idx in range(13):
            pixmaps = {}
            for flip in range(3):
                # Assemble the entire picture
                image = Image.new('RGBA', (128, 512))
                for idx, tile_img in enumerate(self.main_gui.tiles[palette_idx]):
                    if flip & HFLIP:
                        tile_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
                    if flip & VFLIP:
                        tile_img = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
                    x, y = idx % 16, idx // 16
                    image.paste(tile_img, box=(8 * x, 8 * y))
                pixmaps[flip] = QPixmap.fromImage(ImageQt(image))
            self.tile_pixmaps.append(pixmaps)
        self.update_tiles()

    def load_blocks(self):
        """ Reloads the blocks. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.blocks_image = render.draw_blocks(self.main_gui.blocks)
        self.update_blocks()

    def update_blocks(self):
        """ Updates the display of the blocks. """
        self.blocks_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        width, height = self.blocks_image.size
        width, height = int(width * self.zoom_slider.value() / 10), int(height * self.zoom_slider.value() / 10)
        pixmap = QPixmap.fromImage(ImageQt(self.blocks_image)).scaled(width, height)
        self.blocks_scene.addPixmap(pixmap)
        self.blocks_scene.setSceneRect(0, 0, width, height)


    def update_tiles(self):
        """ Updates the display of the tiles widget. """
        self.tiles_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        width, height = int(128 * self.zoom_slider.value() / 10), int(512 * self.zoom_slider.value() / 10)
        flip = (HFLIP if self.tiles_mirror_horizontal_checkbox.isChecked() else 0) | (VFLIP if self.tiles_mirror_vertical_checkbox.isChecked() else 0)
        item = QGraphicsPixmapItem(self.tile_pixmaps[self.tiles_palette_combobox.currentIndex()][flip].scaled(width, height))
        self.tiles_scene.addItem(item)
        self.tiles_scene.setSceneRect(0, 0, width, height)
        
    def zoom_changed(self):
        """ Event handler for when the zoom has changed. """
        self.zoom_label.setText(f'{self.zoom_slider.value() * 10}%')
        self.main_gui.settings['tileset.zoom'] = self.zoom_slider.value()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        self.update_tiles()
        self.update_blocks()


class BlocksScene(QGraphicsScene):
    """ Scene for the individual blocks. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget

    def clear(self):
        super().clear()

class TilesScene(QGraphicsScene):
    """ Scene for the individual tiles. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget

    def clear(self):
        super().clear()

class SelectionScene(QGraphicsScene):
    """ Scene for the selected tiles. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget

    def clear(self):
        super().clear()

class BlockScene(QGraphicsScene):
    """ Scene for the current block. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget

    def clear(self):
        super().clear()


class BlockProperties(ParameterTree):
    """ Tree to display block properties. """

    def __init__(self, tileset_widget, parent=None):
        super().__init__(parent=parent)
        self.tileset_widget = tileset_widget
        self.setHeaderLabels(['Property', 'Value'])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.root = None