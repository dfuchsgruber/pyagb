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
        layout.addWidget(blocks_group, 1, 1, 2, 1)

        current_group = QGroupBox('Current Block')
        current_layout = QGridLayout()
        current_group.setLayout(current_layout)
        layout.addWidget(current_group, 1, 2, 1, 1)

        tiles_group = QGroupBox('Tiles')
        tiles_layout = QGridLayout()
        tiles_group.setLayout(tiles_layout)
        layout.addWidget(tiles_group, 2, 2, 1, 1)

        properties_group = QGroupBox('Properties')
        properties_layout = QGridLayout()
        properties_group.setLayout(properties_layout)
        layout.addWidget(properties_group, 1, 3, 2, 1)

        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)


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