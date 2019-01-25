from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import numpy as np
from PIL.ImageQt import ImageQt
import map_widget, properties, render, blocks, resource_tree, history
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from itertools import product

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
        layout.addWidget(blocks_group, 2, 1, 2, 1)

        seleciton_group = QGroupBox('Selection')
        selection_layout = QGridLayout()
        seleciton_group.setLayout(selection_layout)
        self.selection_scene = SelectionScene(self)
        self.selection_scene_view = QGraphicsView()
        self.selection_scene_view.setViewport(QGLWidget())
        self.selection_scene_view.setScene(self.selection_scene)
        selection_layout.addWidget(self.selection_scene_view)
        layout.addWidget(seleciton_group, 2, 2, 1, 1)

        current_block_group = QGroupBox('Current Block')
        current_block_layout = QGridLayout()
        current_block_group.setLayout(current_block_layout)
        layout.addWidget(current_block_group, 3, 2, 1, 1)

        tiles_group = QGroupBox('Tiles')
        tiles_layout = QGridLayout()
        tiles_group.setLayout(tiles_layout)
        self.tiles_mirror_horizontal_checkbox = QCheckBox('H-Flip')
        tiles_layout.addWidget(self.tiles_mirror_horizontal_checkbox, 1, 2, 1, 1)
        self.tiles_mirror_vertical_checkbox = QCheckBox('V-Flip')
        tiles_layout.addWidget(self.tiles_mirror_vertical_checkbox, 1, 3, 1, 1)
        tiles_palette_group = QGroupBox('Palette')
        tiles_palette_group_layout = QGridLayout()
        tiles_palette_group.setLayout(tiles_palette_group_layout)
        self.tiles_palette_combobox = QComboBox()
        self.tiles_palette_combobox.addItems(map(str, range(13)))
        tiles_palette_group_layout.addWidget(self.tiles_palette_combobox)
        tiles_layout.addWidget(tiles_palette_group, 1, 1, 1, 1)
        self.tiles_scene = TilesScene(self)
        self.tiles_scene_view = QGraphicsView()
        self.tiles_scene_view.setViewport(QGLWidget())
        self.tiles_scene_view.setScene(self.tiles_scene)
        tiles_layout.addWidget(self.tiles_scene_view, 3, 1, 1, 3)
        tiles_layout.setColumnStretch(1, 1)
        tiles_layout.setColumnStretch(2, 0)
        tiles_layout.setColumnStretch(3, 0)
        layout.addWidget(tiles_group, 2, 3, 2, 1)

        zoom_group = QGroupBox('Zoom')
        zoom_layout = QGridLayout()
        zoom_group.setLayout(zoom_layout)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(5)
        self.zoom_slider.setMaximum(30)
        self.zoom_slider.setValue(self.main_gui.settings['tileset.zoom'])
        self.zoom_slider.setTickInterval(1)
        zoom_layout.addWidget(self.zoom_slider, 1, 1, 1, 1)
        self.zoom_label = QLabel(f'{self.zoom_slider.value() * 10}%')
        zoom_layout.addWidget(self.zoom_label, 1, 2, 1, 1)
        layout.addWidget(zoom_group, 1, 1, 1, 3)

        self.info_label = QLabel('')
        layout.addWidget(self.info_label, 4, 1, 1, 3)

        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 0)
        layout.setRowStretch(3, 1)
        layout.setRowStretch(4, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

    def load_header(self):
        """ Updates the blocks of a new header. """


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