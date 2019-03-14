# Widget to display a map and its events

from . import render
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import agb.image
from PIL.ImageQt import ImageQt
import pyqtgraph.parametertree.ParameterTree as ParameterTree
import numpy as np
from skimage.measure import label
from . import properties, history, blocks
import os
import json

class MapWidget(QWidget):
    """ Widget that renders the map and displays properties """

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        # Store blocks in an seperate numpy array that contains the border as well
        self.blocks = None
        self.selection = None
        self.layers = np.array(0)
        self.undo_stack = QUndoStack()
        
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

        # Divide into a widget for blocks and levels
        self.tabs = QTabWidget()
        blocks_widget = QWidget()
        level_widget = QWidget()
        self.tabs.addTab(blocks_widget, 'Blocks')
        self.tabs.addTab(level_widget, 'Level')
        self.tabs.currentChanged.connect(self.tab_changed)

        level_layout = QVBoxLayout()
        level_widget.setLayout(level_layout)
        self.level_opacity_slider = QSlider(Qt.Horizontal)
        self.level_opacity_slider.setMinimum(0)
        self.level_opacity_slider.setMaximum(20)
        self.level_opacity_slider.setSingleStep(1)
        self.level_opacity_slider.setSliderPosition(self.main_gui.settings['map_widget.level_opacity'])
        self.level_opacity_slider.valueChanged.connect(self.change_levels_opacity)
        level_opacity_group = QGroupBox('Opacity')
        level_opactiy_group_layout = QVBoxLayout()
        level_opacity_group.setLayout(level_opactiy_group_layout)
        level_opactiy_group_layout.addWidget(self.level_opacity_slider)
        level_layout.addWidget(level_opacity_group)

        group_selection = QGroupBox('Selection')
        group_selection_layout = QGridLayout()
        group_selection.setLayout(group_selection_layout)
        self.levels_selection_scene = QGraphicsScene()
        self.levels_selection_scene_view = QGraphicsView()
        self.levels_selection_scene_view.setScene(self.levels_selection_scene)
        self.levels_selection_scene_view.setViewport(QGLWidget())
        group_selection_layout.addWidget(self.levels_selection_scene_view, 1, 1, 2, 1)
        level_layout.addWidget(group_selection)

        # Load level gfx
        self.level_blocks_pixmap = QPixmap(os.path.join(os.path.split(__file__)[0], 'level_blocks.png'))
        # And split them
        self.level_blocks_pixmaps = [self.level_blocks_pixmap.copy((idx % 4) * 16, (idx // 4) * 16, 16, 16) for idx in range(0x40)]
        self.level_scene = LevelBlocksScene(self)
        self.level_scene_view = QGraphicsView()
        self.level_scene_view.setViewport(QGLWidget())
        self.level_scene_view.setScene(self.level_scene)
        level_layout.addWidget(self.level_scene_view)
        item = QGraphicsPixmapItem(self.level_blocks_pixmap.scaled(4 * 16 * 2, 16 * 16 * 2))
        self.level_scene.addItem(item)
        item.setAcceptHoverEvents(True)
        item.hoverLeaveEvent = lambda _: self.info_label.setText('')
        self.level_scene.setSceneRect(0, 0, 4 * 16 * 2, 16 * 16 * 2)

        blocks_container = QVBoxLayout()
        blocks_widget.setLayout(blocks_container)
        self.layout.addWidget(self.tabs, 1, 6, 5, 1)
        self.layout.setColumnStretch(1, 4)
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
        self.combo_box_tileset_primary.currentTextChanged.connect(lambda label: self.main_gui.change_tileset(label, True))
        self.combo_box_tileset_secondary.currentTextChanged.connect(lambda label: self.main_gui.change_tileset(label, False))
        group_tileset.setLayout(tileset_layout)

        group_dimensions = QGroupBox('Dimensions')
        blocks_container.addWidget(group_dimensions)
        dimensions_layout = QGridLayout()
        self.label_dimensions = QLabel('[]')
        dimensions_layout.addWidget(self.label_dimensions, 1, 1, 2, 1)
        self.map_change_dimensions = QPushButton('Change Dimensions')
        self.map_change_dimensions.clicked.connect(self.resize_map)
        dimensions_layout.addWidget(self.map_change_dimensions, 1, 2, 2, 1)
        group_dimensions.setLayout(dimensions_layout)
        
        group_border = QGroupBox('Border')
        border_layout = QGridLayout()
        group_border.setLayout(border_layout)
        self.border_scene = BorderScene(self)
        self.border_scene_view = QGraphicsView()
        self.border_scene_view.setViewport(QGLWidget())
        self.border_scene_view.setScene(self.border_scene)
        border_layout.addWidget(self.border_scene_view, 1, 1, 2, 1)
        self.border_change_dimenions = QPushButton('Change Dimensions')
        self.border_change_dimenions.clicked.connect(self.resize_border)
        border_layout.addWidget(self.border_change_dimenions, 1, 2, 1, 1)
        self.show_border = QCheckBox('Show Border')
        self.show_border.setChecked(True)
        self.show_border.toggled.connect(self.load_map)
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
        self.select_levels = QCheckBox('Select Levels')
        self.select_levels.setChecked(False)
        self.select_levels.toggled.connect(self.update_layers)
        group_selection_layout.addWidget(self.select_levels, 3, 1, 1, 1)
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

        self.load_header() # Disables all widgets

    def resize_map(self):
        """ Prompts a resizing of the map. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None: return
        blocks = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_blocks_path'])
        height, width = blocks.shape[0], blocks.shape[1]
        input, ok_pressed = QInputDialog.getText(self, 'Change Map Dimensions', f'Enter new dimensions of footer {self.main_gui.footer_label} in the format "width,height".', text=f'{width},{height}')
        if not ok_pressed: return
        tokens = input.split(',')
        if not len(tokens) == 2:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{input}" is not of format "width,height".')
        try:
            width_new = int(tokens[0].strip(), 0)
        except:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{tokens[0]}" is not a valid width.')
        width_max = self.main_gui.project.config['pymap']['footer']['map_width_max']
        if width_new > width_max or width_new <= 0:
            return QMessageBox.critical(self, 'Invalid Dimeinsions', f'Width {width_new} larger than maximum width {width_max} or non positive')
        try:
            height_new = int(tokens[1].strip(), 0)
        except:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{tokens[1]}" is not a valid height.')
        height_max = self.main_gui.project.config['pymap']['footer']['map_height_max']
        if height_new > height_max or height_new <= 0:
            return QMessageBox.critical(self, 'Invalid Dimeinsions', f'Height {height_new} larger than maximum height {height_max} or non positive')
        self.main_gui.resize_map(height_new, width_new)

    def resize_border(self):
        """ Prompts a resizing of the border. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None: return
        blocks = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['border_path'])
        height, width = blocks.shape[0], blocks.shape[1]
        input, ok_pressed = QInputDialog.getText(self, 'Change Border Dimensions', f'Enter new dimensions of border of footer {self.main_gui.footer_label} in the format "width,height".', text=f'{width},{height}')
        if not ok_pressed: return
        tokens = input.split(',')
        if not len(tokens) == 2:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{input}" is not of format "width,height".')
        try:
            width_new = int(tokens[0].strip(), 0)
        except:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{tokens[0]}" is not a valid width.')
        width_max = self.main_gui.project.config['pymap']['footer']['border_width_max']
        if width_new > width_max:
            return QMessageBox.critical(self, 'Invalid Dimeinsions', f'Width {width_new} larger than maximum width {width_max}')
        try:
            height_new = int(tokens[1].strip(), 0)
        except:
            return QMessageBox.critical(self, 'Invalid Dimensions', f'"{tokens[1]}" is not a valid height.')
        height_max = self.main_gui.project.config['pymap']['footer']['border_height_max']
        if height_new > height_max:
            return QMessageBox.critical(self, 'Invalid Dimeinsions', f'Height {height_new} larger than maximum height {height_max}')
        self.main_gui.resize_border(height_new, width_new)


    def tab_changed(self):
        """ Triggered when the user switches from the blocks to levels tab or vice versa. """
        self.update_layers()
        self.load_map()

    def update_layers(self):
        """ Updates the current layers. """
        if self.tabs.currentIndex() == 0:
            if self.select_levels.isChecked():
                self.layers = np.array([0, 1])
            else:
                self.layers = np.array([0])
        else:
            self.layers = np.array([1])


    def change_levels_opacity(self):
        """ Changes the opacity of the levels. """
        opacity = self.level_opacity_slider.sliderPosition()
        self.main_gui.settings['map_widget.level_opacity'] = opacity
        self.load_map()
        
    def load_project(self, *args):
        """ Update project related widgets. """
        if self.main_gui.project is None: return
        self.combo_box_tileset_primary.blockSignals(True)
        self.combo_box_tileset_primary.clear()
        self.combo_box_tileset_primary.addItems(list(self.main_gui.project.tilesets_primary.keys()))
        self.combo_box_tileset_primary.blockSignals(False)
        self.combo_box_tileset_secondary.blockSignals(True)
        self.combo_box_tileset_secondary.clear()
        self.combo_box_tileset_secondary.addItems(list(self.main_gui.project.tilesets_secondary.keys()))
        self.combo_box_tileset_secondary.blockSignals(False)
        self.set_blocks_selection(np.zeros((1, 1, 2), dtype=np.int))
        self.set_levels_selection(np.zeros((1, 1, 2), dtype=np.int)) 
        self.load_header()

    def load_header(self, *args):
        """ Updates the entire header related widgets. """ 
        # Clear graphics
        self.load_map()
        self.load_border()
        self.load_blocks() 

        if self.main_gui.project is None or self.main_gui.header is None:
            # Reset all widgets
            self.blocks = None
            self.label_dimensions.setText(f'[]')
            self.combo_box_tileset_primary.setEnabled(False)
            self.combo_box_tileset_secondary.setEnabled(False)
            self.show_border.setEnabled(False)
            self.border_change_dimenions.setEnabled(False)
            self.map_change_dimensions.setEnabled(False)
            self.select_levels.setEnabled(False)
        else:
            # Update selection blocks
            self.set_selection(self.selection)  
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
            self.border_change_dimenions.setEnabled(True)
            self.map_change_dimensions.setEnabled(True)
            self.select_levels.setEnabled(True)

    def load_map(self):
        """ Loads the entire map image. """
        self.map_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return
        padded_width, padded_height = self.get_border_padding()
        map_width = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_height_path'])

        # Crop the visible blocks from all blocks including the border
        self.blocks = blocks.compute_blocks(self.main_gui.footer, self.main_gui.project)
        connections = properties.get_member_by_path(self.main_gui.header, self.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        for connection in blocks.filter_visible_connections(blocks.unpack_connections(connections, self.main_gui.project)):
            blocks.insert_connection(self.blocks, connection, self.main_gui.footer, self.main_gui.project)
        visible_width, visible_height = map_width + 2 * padded_width, map_height + 2 * padded_height
        invisible_border_width, invisible_border_height = (self.blocks.shape[1] - visible_width) // 2, (self.blocks.shape[0] - visible_height) // 2
        self.blocks = self.blocks[invisible_border_height : self.blocks.shape[0]-invisible_border_height ,invisible_border_width : self.blocks.shape[1]-invisible_border_width] 

        # Create a pixel map for each block
        self.map_images = np.empty_like(self.blocks[:, :, 0], dtype=object)
        self.level_images = np.empty_like(self.blocks[:, :, 1], dtype=object)
        for (y, x), block_idx in np.ndenumerate(self.blocks[:, :, 0]):
            # Draw the blocks
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
            self.map_images[y, x] = item
        for (y, x), level in np.ndenumerate(self.blocks[:, :, 1]):
            if x in range(padded_width, padded_width + map_width) and y in range(padded_height, padded_height + map_height):
                # Draw the pixmaps
                pixmap = self.level_blocks_pixmaps[level]
                item = QGraphicsPixmapItem(pixmap)
                item.setAcceptHoverEvents(True)
                opacity = QGraphicsOpacityEffect()
                opacity.setOpacity(self.level_opacity_slider.sliderPosition() / 20)
                item.setGraphicsEffect(opacity)
                if self.tabs.currentIndex() == 1: 
                    self.map_scene.addItem(item)
                    item.setPos(16 * x, 16 * y)
                self.level_images[y, x] = item

        # Apply shading to border parts by adding opaque rectangles
        border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['border_color']))
        self.north_border = self.map_scene.addRect(0, 0, (2 * padded_width + map_width) * 16, padded_height * 16, pen = QPen(0), brush = QBrush(border_color))
        self.south_border = self.map_scene.addRect(0, (padded_height + map_height) * 16, (2 * padded_width + map_width) * 16, padded_height * 16, pen = QPen(0), brush = QBrush(border_color))
        self.west_border = self.map_scene.addRect(0, 16 * padded_height, 16 * padded_width, 16 * map_height, pen = QPen(0), brush = QBrush(border_color))
        self.east_border = self.map_scene.addRect(16 * (padded_width + map_width), 16 * padded_height, 16 * padded_width, 16 * map_height, pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.setSceneRect(0, 0, 16 * (2 * padded_width + map_width), 16 * (2 * padded_height + map_height))

    def update_map(self, x, y, layers, blocks):
        """ Updates the map image with new blocks rooted at a certain position. """
        if self.main_gui.project is None or self.main_gui.header is None: return
        padded_width, padded_height = self.get_border_padding()
        map_width = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_height_path'])
        self.blocks[padded_height + y : padded_height + y + blocks.shape[0], padded_width + x : padded_width + x + blocks.shape[1], layers] = blocks[:, :, layers]
        # Redraw relevant pixel maps
        if 0 in layers:
            for (yy, xx), block_idx in np.ndenumerate(blocks[:, :, 0]):
                pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
                self.map_images[padded_height + y + yy, padded_width + x + xx].setPixmap(pixmap)
        if 1 in layers:
            for (yy, xx), level in np.ndenumerate(blocks[:, :, 1]):
                if x + xx in range(map_width) and y + yy in range(map_height):
                    self.level_images[padded_height + y + yy, padded_width + x + xx].setPixmap(self.level_blocks_pixmaps[level])

    def load_blocks(self):
        """ Loads the block pool. """
        self.blocks_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return
        self.blocks_image = QPixmap.fromImage(ImageQt(render.draw_blocks(self.main_gui.blocks)))
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
        self.border_scene.setSceneRect(0, 0, border_blocks.shape[1] * 16, border_blocks.shape[0] * 16)


    def set_selection(self, selection):
        """ Sets currently selected blocks or level blocks depending on the tab. """
        if self.tabs.currentIndex() == 0:
            self.set_blocks_selection(selection)
        elif self.tabs.currentIndex() == 1:
            self.set_levels_selection(selection)
        else:
            raise RuntimeError(f'Unsupported tab {self.tabs.currentIndex() }')

    def set_blocks_selection(self, selection):
        """ Sets currently selected blocks. """
        selection = selection.copy()
        self.selection = selection
        self.selection_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.selection is None: return
        # Block selection
        selection_pixmap = QPixmap.fromImage(ImageQt(render.draw_block_map(self.main_gui.blocks, self.selection)))
        item = QGraphicsPixmapItem(selection_pixmap)
        self.selection_scene.addItem(item)
        self.selection_scene.setSceneRect(0, 0, selection_pixmap.width(), selection_pixmap.height())

    def set_levels_selection(self, selection):
        """ Sets currently selected level blocks. """
        selection = selection.copy()
        self.levels_selection = selection
        self.levels_selection_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.selection is None: return
        # Levels selection
        for (y, x), level in np.ndenumerate(selection[:, :, 1]):
            item = QGraphicsPixmapItem(self.level_blocks_pixmaps[level])
            self.levels_selection_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
        self.levels_selection_scene.setSceneRect(0, 0, selection.shape[1] * 16, selection.shape[0] * 16)

    def get_border_padding(self):
        """ Returns how many blocks are padded to the border of the map. """
        return self.main_gui.project.config['pymap']['display']['border_padding'] if self.show_border.isChecked() else (0, 0)


class BorderScene(QGraphicsScene):
    """ Scene for the border view. """

    def __init__(self, map_widget, parent=None):
        super().__init__(parent=parent)
        self.map_widget = map_widget

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        borders = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['border_path'])
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(borders.shape[1]) and y in range(borders.shape[0]):
            block_idx = borders[y, x, 0]
            self.map_widget.info_label.setText(f'Block : {hex(block_idx)}')
        else:
            return self.map_widget.info_label.setText(f'')

    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        borders = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['border_path'])
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(borders.shape[1]) and y in range(borders.shape[0]) and event.button() == Qt.LeftButton:
            self.map_widget.main_gui.set_border(x, y, self.map_widget.selection)


class LevelBlocksScene(QGraphicsScene):
    """ Scene for the blocks level view. """

    def __init__(self, map_widget, parent=None):
        super().__init__(parent=parent)
        self.map_widget = map_widget

    def mouseMoveEvent(self, event):
        """ Event handler for moving the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        pos = event.scenePos()
        x, y = int(pos.x() / 32), int(pos.y() / 32)
        if x < 0 or x >= 4 or y < 0 or y >= 16:
            return self.map_widget.info_label.setText(f'')
        else:
            self.map_widget.info_label.setText(level_to_info(4 * y + x))
        
    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        pos = event.scenePos()
        x, y = int(pos.x() / 32), int(pos.y() / 32)
        level = 4 * y + x
        if x in range(4) and y in range(16) and (event.button() == Qt.LeftButton or event.button() == Qt.RightButton):
            self.map_widget.set_selection(np.array([[[0, level]]]))

def level_to_info(level):
    """ Computes the info of a level. """
    x, y = level % 4, level // 4
    x_to_collision = {
        0 : 'Passable', 1 : 'Obstacle', 2 : '??? (2)', 3 : '??? (3)'
    }
    if y > 2 and y < 15:
        return f'Level {hex(y)}, {x_to_collision[x]}'
    elif y == 0:
        x_to_collision = {
            0 : 'Connect Levels', 1 : 'Obstacle', 2 : '??? (2)', 3 : '??? (3)'
        }
        return f'{x_to_collision[x]}'
    elif y == 1:
        return f'Water, {x_to_collision[x]}'
    elif y == 15:
        return f'Bridge, {x_to_collision[x]}'

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
        if x < 0 or x >= 8 or y < 0 or y >= 128:
            return self.map_widget.info_label.setText(f'')
        else:
            self.map_widget.info_label.setText(f'Block : {hex(block_idx)}')
        if self.selection_box is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.map_widget.set_selection(render.select_blocks(render.block_map, *self.selection_box))

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
            self.map_widget.set_selection(render.block_map[y:y+1, x:x+1, :])

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
        if x < 0 or x >= 2 * border_width + map_width or y < 0 or y >= 2 * border_height + map_height: return
        #print(x, y, border_width, border_height, map_width, map_height, self.map_widget.blocks.shape)
        if x in range(border_width, border_width + map_width) and y in range(border_height, border_height + map_height):
            if self.map_widget.tabs.currentIndex() == 1:
                # Print the level information
                self.map_widget.info_label.setText(level_to_info(self.map_widget.blocks[y, x, 1]))
            else:
                block = self.map_widget.blocks[y, x]
                self.map_widget.info_label.setText(f'x : {hex(x - border_width)}, y : {hex(y - border_height)}, Block : {hex(block[0])}, Level : {hex(block[1])}')
            # Check if the selection must be drawn
            if not self.last_draw is None and self.last_draw != (x, y):
                self.last_draw = x, y
                map_width = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
                map_height = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
                border_width, border_height = self.map_widget.get_border_padding()
                selection = self.map_widget.selection if self.map_widget.tabs.currentIndex() == 0 else self.map_widget.levels_selection
                self.map_widget.main_gui.set_blocks(x - border_width, y - border_height, self.map_widget.layers, selection)
        else:
            self.map_widget.info_label.setText('')
        if self.selection_box is not None:
            x0, x1, y0, y1 = self.selection_box
            if x1 != x + 1 or y1 != y + 1:
                # Redraw the selection
                self.selection_box = x0, x + 1, y0, y + 1
                self.map_widget.set_selection(render.select_blocks(self.map_widget.blocks, *self.selection_box))

    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.map_widget.main_gui.project is None or self.map_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.map_widget.main_gui.footer, self.map_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        border_width, border_height = self.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x < 0 or x >= 2 * border_width + map_width or y < 0 or y >= 2 * border_height + map_height: return
        if event.button() == Qt.RightButton:
            self.selection_box = x, x + 1, y, y + 1
            self.map_widget.set_selection(self.map_widget.blocks[y:y + 1, x:x + 1])
        elif event.button() == Qt.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                # Replace all blocks of this type with the selection, this is only allowed for 1-block selections
                # Also only one layer is permitted
                layer = self.map_widget.tabs.currentIndex()
                selection = self.map_widget.selection if self.map_widget.tabs.currentIndex() == 0 else self.map_widget.levels_selection
                selection_height, selection_width, _ = selection.shape
                if selection_height == 1 and selection_width == 1 and x in range(border_height, border_width + map_width) and y in range(border_height, border_height + map_height):
                    self.map_widget.main_gui.replace_blocks(x - border_width, y - border_height, layer, selection[0, 0, layer])
            elif modifiers == Qt.ControlModifier:
                # Flood fill is only allowed for 1-block selections
                    # Also only one layer is permitted
                layer = self.map_widget.tabs.currentIndex()
                selection = self.map_widget.selection if self.map_widget.tabs.currentIndex() == 0 else self.map_widget.levels_selection
                selection_height, selection_width, _ = selection.shape
                if selection_height == 1 and selection_width == 1 and x in range(border_height, border_width + map_width) and y in range(border_height, border_height + map_height):
                    self.map_widget.main_gui.flood_fill(x - border_width, y - border_height, layer, selection[0, 0, layer])
            else:
                self.last_draw = -1, -1 # This triggers the drawing routine
                self.map_widget.undo_stack.beginMacro('Drawing Blocks')
                self.mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.RightButton:
            self.selection_box = None
        if event.button() == Qt.LeftButton:
            self.last_draw = None
            #self.map_widget.history.close()
            self.map_widget.undo_stack.endMacro()



