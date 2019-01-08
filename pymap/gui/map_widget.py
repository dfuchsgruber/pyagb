# Widget to display a map and its events

import render
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import agb.image
from PIL.ImageQt import ImageQt
import pyqtgraph.parametertree.ParameterTree as ParameterTree
import numpy as np
import properties

class MapWidget(QWidget):
    """ Widget that renders the map and displays properties """

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        # Store blocks in an seperate numpy array that contains the border as well
        self.blocks = None
        self.selection = None

        # (Re-)Build the ui
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.map_scene = MapScene(self)
        self.map_scene_view = QGraphicsView()
        self.map_scene_view.setViewport(QGLWidget())
        self.map_scene_view.setScene(self.map_scene)
        self.layout.addWidget(self.map_scene_view, 1, 1, 5, 5)

        self.info_label = QLabel()
        self.layout.addWidget(self.info_label, 6, 1, 1, 6)
        self.layout.setRowStretch(1, 5)
        self.layout.setRowStretch(6, 0)

        blocks_container = QVBoxLayout()
        self.layout.addLayout(blocks_container, 1, 6, 5, 1)
        self.layout.setColumnStretch(1, 5)
        self.layout.setColumnStretch(6, 1)

        group_tileset = QGroupBox('Tileset')
        blocks_container.addWidget(group_tileset)
        tileset_layout = QGridLayout()
        tileset_layout.addWidget(QLabel('Primary'), 1, 1)
        tileset_layout.addWidget(QLabel('Secondary'), 2, 1)

        self.combo_box_tileset_primary = QComboBox()
        tileset_layout.addWidget(self.combo_box_tileset_primary, 1, 2)
        self.combo_box_tileset_secondary = QComboBox()
        tileset_layout.addWidget(self.combo_box_tileset_secondary, 2, 2)
        self.combo_box_tileset_primary.currentTextChanged.connect(lambda label: self.main_gui.open_tilesets(label_primary=label))
        self.combo_box_tileset_secondary.currentTextChanged.connect(lambda label: self.main_gui.open_tilesets(label_secondary=label))
        group_tileset.setLayout(tileset_layout)

        group_dimensions = QGroupBox('Dimensions')
        blocks_container.addWidget(group_dimensions)
        dimensions_layout = QGridLayout()
        self.label_dimensions = QLabel('[]')
        dimensions_layout.addWidget(self.label_dimensions, 1, 1, 2, 1)
        map_change_dimensions = QPushButton('Change Dimensions')
        dimensions_layout.addWidget(map_change_dimensions, 1, 2, 2, 1)
        group_dimensions.setLayout(dimensions_layout)
        
        group_border = QGroupBox('Border')
        border_layout = QGridLayout()
        group_border.setLayout(border_layout)
        self.border_scene = QGraphicsScene()
        self.border_scene_view = QGraphicsView()
        self.border_scene_view.setViewport(QGLWidget())
        self.border_scene_view.setScene(self.border_scene)
        border_layout.addWidget(self.border_scene_view, 1, 1, 2, 1)
        border_change_dimenions = QPushButton('Change Dimensions')
        border_layout.addWidget(border_change_dimenions, 1, 2, 1, 1)
        self.show_border = QCheckBox('Show Border')
        self.show_border.setChecked(True)
        self.show_border.toggled.connect(self.load_header)
        border_layout.addWidget(self.show_border, 2, 2, 1, 1)
        blocks_container.addWidget(group_border)

        group_selection = QGroupBox('Selection')
        group_selection_layout = QGridLayout()
        group_selection.setLayout(group_selection_layout)
        self.selection_scene = QGraphicsScene()
        self.selection_scene_view = QGraphicsView()
        self.selection_scene_view.setScene(self.selection_scene)
        self.selection_scene_view.setViewport(QGLWidget())
        group_selection_layout.addWidget(self.selection_scene_view, 1, 1, 2, 1)
        blocks_container.addWidget(group_selection)
    
        group_blocks = QWidget()
        group_blocks_layout = QGridLayout()
        group_blocks.setLayout(group_blocks_layout)
        self.blocks_scene = BlocksScene(self)
        self.blocks_scene_view = QGraphicsView()
        self.blocks_scene_view.setViewport(QGLWidget())
        self.blocks_scene_view.setScene(self.blocks_scene)
        group_blocks_layout.addWidget(self.blocks_scene_view)
        group_blocks.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        blocks_container.addWidget(group_blocks)

    def load_project(self, *args):
        """ Update project related widgets. """
        if self.main_gui.project is None: returnself.combo_box_tileset_primary.blockSignals(True)
        self.combo_box_tileset_primary.clear()
        self.combo_box_tileset_primary.blockSignals(False)
        self.combo_box_tileset_secondary.blockSignals(True)
        self.combo_box_tileset_secondary.clear()
        self.combo_box_tileset_secondary.blockSignals(False)
        self.combo_box_tileset_primary.addItems(list(self.main_gui.project.tilesets_primary.keys()))
        self.combo_box_tileset_secondary.addItems(list(self.main_gui.project.tilesets_secondary.keys()))


    def load_header(self, *args):
        """ Updates the entire header related widgets. """ 
        # Clear graphics
        self.load_map()
        self.load_border()
        self.load_blocks()
        self.set_selection(np.array([[{'block_idx' : 0, 'level' : 0}]]))      

        if self.main_gui.project is None or self.main_gui.header is None:
            # Reset all widgets
            self.blocks = None
            self.label_dimensions.setText(f'[]')
            self.combo_box_tileset_primary.setEnabled(False)
            self.combo_box_tileset_secondary.setEnabled(False)
            self.show_border.setEnabled(False)
        else:
            self.combo_box_tileset_primary.setEnabled(True)
            self.combo_box_tileset_secondary.setEnabled(True)
            self.show_border.setEnabled(True) 
            map_width = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_width_path'])
            map_height = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_height_path'])
            self.label_dimensions.setText(f'[{map_width}, {map_height}]')
            tileset_primary_label = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['tileset_primary_path'])
            tileset_secondary_label = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['tileset_secondary_path'])
            self.combo_box_tileset_primary.blockSignals(True)
            self.combo_box_tileset_primary.setCurrentText(tileset_primary_label)
            self.combo_box_tileset_primary.blockSignals(False)
            self.combo_box_tileset_secondary.blockSignals(True)
            self.combo_box_tileset_secondary.setCurrentText(tileset_secondary_label)
            self.combo_box_tileset_secondary.blockSignals(False)

    def load_map(self):
        """ Loads the entire map image. """
        self.map_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return

        map_blocks = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_blocks_path'])
        map_height, map_width = map_blocks.shape
        padded_width, padded_height = self.get_border_padding()
        border_blocks = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['border_path'])
        border_height, border_width = border_blocks.shape

        # The border is always aligned with the map, therefore one has to consider a virtual block array that is larger than what acutally is displayed
        virtual_reps_x = (int(np.ceil(padded_width / border_width)) + int(np.ceil((map_width + padded_width) / border_width)))
        virtual_reps_y = (int(np.ceil(padded_height / border_height)) + int(np.ceil((map_height + padded_height) / border_height)))
        self.blocks = np.tile(border_blocks, (virtual_reps_y, virtual_reps_x))
        x0, y0 = (border_width - (padded_width % border_width)) % border_width, (border_height - (padded_height % border_height)) % border_height
        # Create frame exactly the size of the map and its borders repeated with the border sequence, aligned with the origin of the map
        self.blocks = self.blocks[y0:y0 + map_height + 2 * padded_height, x0:x0 + map_width + 2 * padded_width]
        # Insert the map into this frame
        self.blocks[padded_height:padded_height + map_height, padded_width:padded_width + map_width] = map_blocks

        # Create a pixel map for each block
        self.map_images = np.empty_like(self.blocks, dtype=object)
        for (y, x), block in np.ndenumerate(self.blocks):
            block_idx = block['block_idx']
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
            self.map_images[y, x] = item

        # Apply shading to border parts by adding opaque rectangles
        border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['border_color']))
        north = self.map_scene.addRect(0, 0, (2 * padded_width + map_width) * 16, padded_height * 16, pen = QPen(0), brush = QBrush(border_color))
        south = self.map_scene.addRect(0, (padded_height + map_height) * 16, (2 * padded_width + map_width) * 16, padded_height * 16, pen = QPen(0), brush = QBrush(border_color))
        west = self.map_scene.addRect(0, 16 * padded_height, 16 * padded_width, 16 * map_height, pen = QPen(0), brush = QBrush(border_color))
        east = self.map_scene.addRect(16 * (padded_width + map_width), 16 * padded_height, 16 * padded_width, 16 * map_height, pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.setSceneRect(0, 0, 16 * (2 * padded_width + map_width), 16 * (2 * padded_height + map_height))

    def update_map(self, x, y, blocks):
        """ Updates the map image with new blocks rooted at a certain position. """
        if self.main_gui.project is None or self.main_gui.header is None: return
        padded_width, padded_height = self.get_border_padding()
        self.blocks[padded_height + y : padded_height + y + blocks.shape[0], padded_width + x : padded_width + x + blocks.shape[1]] = blocks
        # Redraw relevant pixel maps
        for (yy, xx), block in np.ndenumerate(blocks):
            block_idx = block['block_idx']
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            self.map_images[padded_height + y + yy, padded_width + x + xx].setPixmap(pixmap)

    def load_blocks(self):
        """ Loads the block pool. """
        self.blocks_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return
        self.blocks_image = QPixmap.fromImage(ImageQt(render.draw_block_map(
            self.main_gui.blocks, BLOCK_MAP[:len(self.main_gui.blocks)])))
        item = QGraphicsPixmapItem(self.blocks_image)
        self.blocks_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        item.hoverLeaveEvent = lambda _: self.info_label.setText('')

    def load_border(self):
        """ Loads the border. """
        self.border_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return
        border_blocks = np.array(properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['border_path']))
        self.border_image = QPixmap.fromImage(ImageQt(render.draw_block_map(self.main_gui.blocks, border_blocks)))
        self.border_scene.addPixmap(self.border_image)


    def set_selection(self, selection):
        """ Sets currently selected blocks. """
        self.selection = selection
        self.selection_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.selection is None: return
        selection_pixmap = QPixmap.fromImage(ImageQt(render.draw_block_map(self.main_gui.blocks, self.selection)))
        item = QGraphicsPixmapItem(selection_pixmap)
        self.selection_scene.addItem(item)
        self.selection_scene.setSceneRect(0, 0, selection_pixmap.width(), selection_pixmap.height())

    def get_border_padding(self):
        """ Returns how many blocks are padded to the border of the map. """
        return self.main_gui.project.config['pymap']['display']['border_padding'] if self.show_border.isChecked() else (0, 0)




class BlocksScene(QGraphicsScene):
    """ Scene for the blocks view. """

    def __init__(self, map_widget, parent=None):
        super().__init__(parent=parent)
        self.map_widget = map_widget
        self.selection_box = None

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        block_idx = 8 * y + x
        self.map_widget.info_label.setText(f'Block : {hex(block_idx)}')
        if self.selection_box is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.map_widget.set_selection(select_blocks(BLOCK_MAP, *self.selection_box))

    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.RightButton:
            self.selection_box = None

    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if event.button() == Qt.RightButton:
            self.selection_box = x, x + 1, y, y + 1
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            # Select the current block
            self.map_widget.set_selection(BLOCK_MAP[y:y+1, x:x+1])

class MapScene(QGraphicsScene):
    """ Scene for the map view. """

    def __init__(self, map_widget, parent=None):
        super().__init__(parent=parent)
        self.map_widget = map_widget
        self.selection_box = None
        self.last_draw = None # Store the position where a draw happend recently so there are not multiple draw events per block

    def mouseMoveEvent(self, event):
        """ Event handler for hover events on the map image. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        border_width, border_height = self.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        #print(x, y, border_width, border_height, map_width, map_height, self.map_widget.blocks.shape)
        if x in range(border_width, border_width + map_width) and y in range(border_height, border_height + map_height):
            block = self.map_widget.blocks[y, x]
            self.map_widget.info_label.setText(f'x : {hex(x - border_width)}, y : {hex(y - border_height)}, Block : {hex(block["block_idx"])}, Level : {hex(block["level"])}')
            # Check if the selection must be drawn
            if not self.last_draw is None and self.last_draw != (x, y):
                self.last_draw = x, y
                map_width = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
                map_height = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
                border_width, border_height = self.map_widget.get_border_padding()
                self.map_widget.main_gui.set_blocks(x - border_width, y - border_height, self.map_widget.selection)
        else:
            self.map_widget.info_label.setText('')
        if self.selection_box is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.map_widget.set_selection(select_blocks(self.map_widget.blocks, *self.selection_box))

    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if event.button() == Qt.RightButton:
            self.selection_box = x, x + 1, y, y + 1
            self.map_widget.set_selection(self.map_widget.blocks[y:y + 1, x:x + 1])
        elif event.button() == Qt.LeftButton:
            self.last_draw = -1, -1 # This triggers the drawing routine
            self.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.RightButton:
            self.selection_box = None
        if event.button() == Qt.LeftButton:
            self.last_draw = None

# Static block map for the blocks widget
BLOCK_MAP = np.array([{'block_idx' : idx, 'level' : 0} for idx in range(0x400)]).reshape((-1, 8))

def select_blocks(blocks, x0, x1, y0, y1):
    """ Helper method to select a subset of blocks by a box. """
    if x1 <= x0:
        x0, x1 = x1 - 1, x0 + 1
    if y1 <= y0:
        y0, y1 = y1 - 1, y0 + 1
    return blocks[y0:y1, x0:x1]