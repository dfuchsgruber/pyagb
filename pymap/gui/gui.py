# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys, os
import json
from copy import deepcopy
import numpy as np
from skimage.measure import label
import appdirs
import resource_tree, map_widget, footer_widget, properties, render, history, header_widget, event_widget
import pymap.project
from settings import Settings

HISTORY_SET_BLOCKS = 0

class PymapGui(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = Settings()
        self.project = None
        self.project_path = None
        self.header = None
        self.header_bank = None
        self.header_map_idx = None
        self.footer = None
        self.footer_label = None
        self.tileset_primary = None
        self.tileset_primary_label = None
        self.tileset_secondary = None
        self.tileset_secondary_label = None

        # Central storage for blocks and tiles, subwidget access it via parent references
        self.blocks = None
        self.tiles = None

        self.setup_ui()

    def setup_ui(self):

        # Add the project tree widget
        self.resource_tree_widget = QDockWidget('Project Resources')
        self.resource_tree = resource_tree.ResourceParameterTree(self)
        self.resource_tree_widget.setWidget(self.resource_tree)
        self.resource_tree_widget.setFloating(False)
        self.resource_tree_widget.setFeatures(QDockWidget.DockWidgetFloatable | 
                 QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.resource_tree_widget)

        # Add the tabs 
        self.central_widget = QTabWidget()
        self.map_widget = map_widget.MapWidget(self)
        self.event_widget = event_widget.EventWidget(self)
        self.header_widget = header_widget.HeaderWidget(self)
        self.footer_widget = footer_widget.FooterWidget(self)
        self.tileset_widget = QTextEdit()
        self.central_widget.addTab(self.map_widget, 'Map')
        self.central_widget.addTab(self.event_widget, 'Events')
        self.central_widget.addTab(self.tileset_widget, 'Tileset')
        self.central_widget.addTab(self.header_widget, 'Header')
        self.central_widget.addTab(self.footer_widget, 'Footer')
        self.central_widget.currentChanged.connect(self.tab_changed)

        # Build the menu bar
        # 'File' menu
        file_menu = self.menuBar().addMenu('&File')
        # 'New' submenu
        file_menu_new_menu = file_menu.addMenu('&New')
        file_menu_new_project_action = file_menu_new_menu.addAction('Project')
        file_menu_new_project_action.setShortcut('Ctrl+N')
        file_menu_new_bank_action = file_menu_new_menu.addAction('Bank')
        file_menu_new_bank_action.triggered.connect(self.resource_tree.create_bank)
        file_menu_new_header_action = file_menu_new_menu.addAction('Header')
        file_menu_new_header_action.triggered.connect(self.resource_tree.create_header)
        file_menu_new_footer_action = file_menu_new_menu.addAction('Footer')
        file_menu_new_footer_action.triggered.connect(self.resource_tree.create_footer)
        file_menu_new_tileset_action = file_menu_new_menu.addAction('Tileset')
        file_menu_new_tileset_action.triggered.connect(self.resource_tree.create_tileset)
        # Flat actions
        file_menu_open_action = file_menu.addAction('&Open Project')
        file_menu_open_action.triggered.connect(self.open_project)
        file_menu_open_action.setShortcut('Ctrl+O')
        # 'Save' submenu
        file_menu_save_menu = file_menu.addMenu('&Save')
        file_menu_save_all = file_menu_save_menu.addAction('All')
        file_menu_save_all.triggered.connect(self.save_all)
        file_menu_save_all.setShortcut('Ctrl+S')
        file_menu_save_project = file_menu_save_menu.addAction('Project')
        file_menu_save_project.triggered.connect(self.save_project)
        file_menu_save_header = file_menu_save_menu.addAction('Header')
        file_menu_save_header.triggered.connect(self.save_header)
        file_menu_save_footer = file_menu_save_menu.addAction('Footer')
        file_menu_save_footer.triggered.connect(self.save_footer)
        file_menu_save_tilesets = file_menu_save_menu.addAction('Tilesets')
        # 'Edit' menu
        edit_menu = self.menuBar().addMenu('&Edit')
        edit_menu_undo_action = edit_menu.addAction('Undo')
        edit_menu_undo_action.triggered.connect(lambda: self.central_widget.currentWidget().undo_stack.undo())
        edit_menu_undo_action.setShortcut('Ctrl+Z')
        edit_menu_redo_action = edit_menu.addAction('Redo')
        edit_menu_redo_action.triggered.connect(lambda: self.central_widget.currentWidget().undo_stack.redo())
        edit_menu_redo_action.setShortcut('Ctrl+Y')
        # 'View' menu
        view_menu = self.menuBar().addMenu('&View')
        view_menu_resource_action = view_menu.addAction('Toggle Header Listing')
        view_menu_resource_action.setShortcut('Ctrl+L')
        view_menu_resource_action.triggered.connect(self.resource_tree_toggle_header_listing)

        self.setCentralWidget(self.central_widget)

    def tab_changed(self):
        """ Callback method for when a tab is changed. """
        if self.central_widget.currentWidget() is self.event_widget:
            self.event_widget.load_map() # The updates to the map are lazy: Only update the map when the tab is opened.

    def save_all(self):
        """ Saves project, header, footer and tilesets. """
        self.save_project()
        self.save_header()
        self.save_footer()

    def save_project(self):
        """ Saves the current project. """
        if self.project is None: return
        self.project.save(self.project_path)

    def open_project(self):
        """ Prompts a dialog to open a new project file. """
        path, suffix = QFileDialog.getOpenFileName(self, 'Open project', self.settings['recent.project'], 'Pymap projects (*.pmp)')
        if len(path):
            os.chdir(os.path.dirname(path))
            self.project_path = path
            self.settings['recent.project'] = path
            self.project = pymap.project.Project(path)
            self.resource_tree.load_project()
            self.map_widget.load_project()
            self.footer_widget.load_project()
            self.header_widget.load_project()
            self.event_widget.load_project()

    def clear_header(self):
        """ Unassigns the current header, footer, tilesets. """
        self.header = None
        self.header_bank = None
        self.header_map_idx = None
        # Render subwidgets
        self.map_widget.load_header()

    def open_header(self, bank, map_idx):
        """ Opens a new map header and displays it. """
        if self.project is None: return
        # Check if the history of the map or header needs to be saved
        if self.header is not None and (not self.header_widget.undo_stack.isClean()): # TODO: Event stack
            pressed = QMessageBox.question(self, 'Save Header Changes', f'Header {self.project.headers[self.header_bank][self.header_map_idx][0]} has changed. Do you want to save changes?')
            if pressed == QMessageBox.Yes: self.save_header()
            if pressed != QMessageBox.Yes and pressed != QMessageBox.No: return # Stay on map 
        self.header_widget.undo_stack.clear() # TODO: Event stack
        os.chdir(os.path.dirname(self.project_path))
        label, path, namespace = self.project.headers[bank][map_idx]
        with open(path, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['header']['datatype'])
        assert(content['label'] == label)
        self.header = content['data']
        self.header_bank = bank
        self.header_map_idx = map_idx
        # Trigger opening of the fooster
        footer_label = properties.get_member_by_path(self.header, self.project.config['pymap']['header']['footer_path'])
        self.open_footer(footer_label)

    def open_footer(self, label):
        """ Opens a new footer and assigns it to the current header. """
        if self.project is None or self.header is None: return
        # Check if the history of the map or footer needs to be saved
        if self.footer is not None and (not self.map_widget.undo_stack.isClean() or not self.footer_widget.undo_stack.isClean()):
            pressed = QMessageBox.question(self, 'Save Footer Changes', f'Footer {self.footer_label} has changed. Do you want to save changes?')
            if pressed == QMessageBox.Yes: self.save_footer()
            if pressed != QMessageBox.Yes and pressed != QMessageBox.No: return # Stay on map 
        self.map_widget.undo_stack.clear()
        self.footer_widget.undo_stack.clear()
        os.chdir(os.path.dirname(self.project_path))
        footer_idx, path = self.project.footers[label]
        with open(path, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['footer']['datatype'])
        assert(content['label'] == label)
        self.footer = content['data']
        self.footer_label = label
        properties.set_member_by_path(self.header, label, self.project.config['pymap']['header']['footer_path'])
        properties.set_member_by_path(self.header, footer_idx, self.project.config['pymap']['header']['footer_idx_path'])
        # Accelerate computiations by storing map blocks and borders in numpy arrays
        map_blocks = map_widget.blocks_to_ndarray(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path']))
        properties.set_member_by_path(self.footer, map_blocks, self.project.config['pymap']['footer']['map_blocks_path'])
        border_blocks = map_widget.blocks_to_ndarray(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['border_path']))
        properties.set_member_by_path(self.footer, border_blocks, self.project.config['pymap']['footer']['border_path'])
        # Trigger opening the tilesets
        tileset_primary_label = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_primary_path'])
        tileset_secondary_label = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_secondary_path'])
        self.open_tilesets(tileset_primary_label, tileset_secondary_label)

    def open_tilesets(self, label_primary=None, label_secondary=None):
        """ Opens and assigns a new primary tileset and secondary tileset to the current footer. """
        if self.project is None or self.header is None or self.footer is None: return
        os.chdir(os.path.dirname(self.project_path))
        if label_primary is not None:
            # If the footer is assigned a null reference, do not render
            path_primary = self.project.tilesets_primary[label_primary]
            with open(path_primary, encoding=self.project.config['json']['encoding']) as f:
                content = json.load(f)
            assert(content['type'] == self.project.config['pymap']['tileset_primary']['datatype'])
            assert(content['label'] == label_primary)
            self.tileset_primary = content['data']
            self.tileset_primary_label = label_primary
            properties.set_member_by_path(self.footer, label_primary, self.project.config['pymap']['footer']['tileset_primary_path'])
        if label_secondary is not None:
            path_secondary = self.project.tilesets_secondary[label_secondary]
            with open(path_secondary, encoding=self.project.config['json']['encoding']) as f:
                content = json.load(f)
            assert(content['type'] == self.project.config['pymap']['tileset_secondary']['datatype'])
            assert(content['label'] == label_secondary)
            self.tileset_secondary = content['data']
            self.tileset_secondary_label = label_secondary
            properties.set_member_by_path(self.footer, label_secondary, self.project.config['pymap']['footer']['tileset_secondary_path'])
        if label_primary is not None or label_secondary is not None:
        # Load the gfx and render tiles
            self.tiles = render.get_tiles(self.tileset_primary, self.tileset_secondary, self.project)
            self.blocks = render.get_blocks(self.tileset_primary, self.tileset_secondary, self.tiles, self.project)
            self._update()

    def save_footer(self):
        """ Saves the current map footer. """
        if self.project is None or self.header is None or self.footer is None: return
        # Convert blocks and borders back to lists
        footer = deepcopy(self.footer)
        map_blocks = map_widget.ndarray_to_blocks(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path']))
        properties.set_member_by_path(footer, map_blocks, self.project.config['pymap']['footer']['map_blocks_path'])
        border_blocks = map_widget.ndarray_to_blocks(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['border_path']))
        properties.set_member_by_path(footer, border_blocks, self.project.config['pymap']['footer']['border_path'])
        footer_idx, path = self.project.footers[self.footer_label]
        with open(path, 'w+', encoding=self.project.config['json']['encoding']) as f:
            json.dump(
                {'type' : self.project.config['pymap']['footer']['datatype'], 'label' : self.footer_label, 'data' : footer}, f, 
                indent=self.project.config['json']['indent'])
        # Adapt history
        self.map_widget.undo_stack.setClean()
        self.footer_widget.undo_stack.setClean()

    def save_header(self):
        """ Saves the current map header.  """
        if self.project is None or self.header is None: return
        label, path, _ = self.project.headers[self.header_bank][self.header_map_idx]
        with open(path, 'w+', encoding=self.project.config['json']['encoding']) as f:
            json.dump(
                {'type' : self.project.config['pymap']['header']['datatype'], 'label' : label, 'data' : self.header}, f, 
                indent=self.project.config['json']['indent'])
        # Adapt history
        self.header_widget.undo_stack.setClean()
        # TODO: Event stack

        
    def _update(self):
        self.map_widget.load_header()
        self.footer_widget.load_footer()
        self.header_widget.load_header()
        self.event_widget.load_header() # It is important to place this after the map widget, since it reuses its tiling
        # TODO: other widgets
        pass

    def resource_tree_toggle_header_listing(self):
        """ Toggles the listing method for the resource tree. """
        if self.project is None: return
        self.settings['resource_tree.header_listing'] = (
            resource_tree.SORT_BY_BANK if self.settings['resource_tree.header_listing'] == resource_tree.SORT_BY_NAMESPACE else
            resource_tree.SORT_BY_NAMESPACE
        )
        self.resource_tree.load_headers()

    def set_border(self, x, y, blocks):
        """ Sets the blocks of the border and adds an action to the history.  """
        border = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['border_path'])
        window = border[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[:window.shape[0], :window.shape[1]].copy()
        self.map_widget.undo_stack.push(history.SetBorder(self, x, y, blocks, window))

    def set_blocks(self, x, y, layers, blocks):
        """ Sets the blocks on the header and adds an item to the history. """
        if self.project is None or self.header is None: return
        map_blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])
        # Truncate blocks to fit the map
        window = map_blocks[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[:window.shape[0], :window.shape[1]].copy()
        self.map_widget.undo_stack.push(history.SetBlocks(self, x, y, layers, blocks, window))

    def flood_fill(self, x, y, layer, value):
        """ Flood fills with origin (x, y) and a certain layer with a new value. """
        if self.project is None or self.header is None: return
        map_blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])[:, :, layer]
        labeled = label(map_blocks + 1, connectivity=1) # Seems like 0 is not recognized by the connectivity
        idx = np.where(labeled == labeled[y, x])
        self.map_widget.undo_stack.push(history.ReplaceBlocks(self, idx, layer, value, map_blocks[y, x]))
        
    def replace_blocks(self, x, y, layer, value):
        """ Replaces all blocks that are like (x, y) w.r.t. to the layer by the new value. """
        if self.project is None or self.header is None: return
        map_blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])[:, :, layer]
        idx = np.where(map_blocks == map_blocks[y, x])
        self.map_widget.undo_stack.push(history.ReplaceBlocks(self, idx, layer, value, map_blocks[y, x]))

    def resize_map(self, height_new, width_new):
        """ Changes the map dimensions. """
        blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])
        height, width = blocks.shape[0], blocks.shape[1]
        if height != height_new or width != width_new:
            self.map_widget.undo_stack.push(history.ResizeMap(self, height_new, width_new, blocks))

    def resize_border(self, height_new, width_new):
        """ Changes the border dimensions. """
        blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['border_path'])
        height, width = blocks.shape[0], blocks.shape[1]
        if height != height_new or width != width_new:
            self.map_widget.undo_stack.push(history.ResizeBorder(self, height_new, width_new, blocks))

    def change_tileset(self, label, primary):
        """ Changes the current tileset by performing a command. """
        if self.project is None or self.header is None or self.footer is None: return
        label_old = self.tileset_primary_label if primary else self.tileset_secondary_label
        self.map_widget.undo_stack.push(history.AssignTileset(self, primary, label, label_old))

        
def main():
    #os.chdir('/media/d/romhacking/Violet_Sources')
    app = QApplication(sys.argv)
    ex = PymapGui()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()