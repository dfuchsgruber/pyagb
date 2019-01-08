# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys, os
import json
import numpy as np
import appdirs
import resource_tree, map_widget, properties, render
import pymap.project
from settings import Settings

HISTORY_SET_BLOCKS = 0

class PymapGui(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = Settings()
        self.project = None
        self.header = None
        self.header_label = None
        self.footer = None
        self.footer_label = None
        self.tileset_primary = None
        self.tileset_primary_label = None
        self.tileset_secondary = None
        self.tileset_secondary_label = None
        self.history = []
        self.history_idx = 0

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
        self.level_widget = QTextEdit()
        self.header_widget = QTextEdit()
        self.footer_widget = QTextEdit()
        self.tileset_widget = QTextEdit()
        self.central_widget.addTab(self.map_widget, 'Map')
        self.central_widget.addTab(self.level_widget, 'Level')
        self.central_widget.addTab(self.header_widget, 'Header')
        self.central_widget.addTab(self.footer_widget, 'Footer')
        self.central_widget.addTab(self.tileset_widget, 'Tileset')

        # Build the menu bar
        # 'File' menu
        self.file_menu = self.menuBar().addMenu('&File')
        # 'New' submenu
        self.file_menu_new_menu = self.file_menu.addMenu('&New')
        self.file_menu_new_project_action = self.file_menu_new_menu.addAction('Project')
        self.file_menu_new_project_action.setShortcut('Ctrl+N')
        self.file_menu_new_bank_action = self.file_menu_new_menu.addAction('Bank')
        self.file_menu_new_bank_action.triggered.connect(self.resource_tree.create_bank)
        self.file_menu_new_header_action = self.file_menu_new_menu.addAction('Header')
        self.file_menu_new_header_action.triggered.connect(self.resource_tree.create_header)
        self.file_menu_new_footer_action = self.file_menu_new_menu.addAction('Footer')
        self.file_menu_new_footer_action.triggered.connect(self.resource_tree.create_footer)
        self.file_menu_new_tileset_action = self.file_menu_new_menu.addAction('Tileset')
        self.file_menu_new_tileset_action.triggered.connect(self.resource_tree.create_tileset)
        # Flat actions
        self.file_menu_open_action = self.file_menu.addAction('&Open Project')
        self.file_menu_open_action.setShortcut('Ctrl+O')
        self.file_menu_open_action.triggered.connect(self.open_project)
        self.file_menu_save_action = self.file_menu.addAction('&Save Project')
        self.file_menu_save_action.setShortcut('Ctrl+S')
        # 'Edit' menu
        self.edit_menu = self.menuBar().addMenu('&Edit')
        self.edit_menu_undo_action = self.edit_menu.addAction('Undo')
        self.edit_menu_undo_action.triggered.connect(self._undo)
        self.edit_menu_undo_action.setShortcut('Ctrl+Z')
        self.edit_menu_redo_action = self.edit_menu.addAction('Redo')
        self.edit_menu_redo_action.triggered.connect(self._redo)
        self.edit_menu_redo_action.setShortcut('Ctrl+Y')
        # 'View' menu
        self.view_menu = self.menuBar().addMenu('&View')
        self.view_menu_resource_action = self.view_menu.addAction('Toggle Header Listing')
        self.view_menu_resource_action.setShortcut('Ctrl+L')
        self.view_menu_resource_action.triggered.connect(self.resource_tree_toggle_header_listing)

        self.setCentralWidget(self.central_widget)

    def open_project(self):
        """ Prompts a dialog to open a new project file. """
        path, suffix = QFileDialog.getOpenFileName(self, 'Open project', self.settings['recent.project'], 'Pymap projects (*.pmp)')
        if len(path):
            os.chdir(os.path.dirname(path))
            self.settings['recent.project'] = path
            self.project = pymap.project.Project(path)
            self.resource_tree.load_project()
            self.map_widget.load_project()
            self.history = []
            self.history_idx = 0

    def clear_header(self):
        """ Unassigns the current header, footer, tilesets. """
        self.header = None
        self.header_label = None
        # Render subwidgets
        self.map_widget.load_header()

    def open_header(self, bank, map_idx):
        """ Opens a new map header and displays it. """
        if self.project is None: return
        label, path, namespace = self.project.headers[bank][map_idx]
        with open(path, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['header']['datatype'])
        assert(content['label'] == label)
        self.header = content['data']
        self.header_label = label
        # Trigger opening of the footer
        footer_label = properties.get_member_by_path(self.header, self.project.config['pymap']['header']['footer_path'])
        self.open_footer(footer_label)

    def open_footer(self, label):
        """ Opens a new footer and assigns it to the current header. """
        if self.project is None or self.header is None: return
        footer_idx, path = self.project.footers[label]
        with open(path, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['footer']['datatype'])
        assert(content['label'] == label)
        self.footer = content['data']
        self.footer_label = label
        properties.set_member_by_path(self.header, label, self.project.config['pymap']['header']['footer_path'])
        # Accelerate computiations by storing map blocks and borders in numpy arrays
        map_blocks = np.array(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path']))
        properties.set_member_by_path(self.footer, map_blocks, self.project.config['pymap']['footer']['map_blocks_path'])
        border_blocks = np.array(properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['border_path']))
        properties.set_member_by_path(self.footer, border_blocks, self.project.config['pymap']['footer']['border_path'])
        # Trigger opening the tilesets
        tileset_primary_label = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_primary_path'])
        tileset_secondary_label = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_secondary_path'])
        self.open_tilesets(tileset_primary_label, tileset_secondary_label)

    def open_tilesets(self, label_primary=None, label_secondary=None):
        """ Opens and assigns a new primary tileset and secondary tileset to the current footer. """
        if self.project is None or self.header is None or self.footer is None: return
        if label_primary is None: label_primary = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_primary_path'])
        if label_secondary is None: label_secondary = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['tileset_secondary_path'])
        # If the footer is assigned a null reference, do not render
        path_primary = self.project.tilesets_primary[label_primary]
        with open(path_primary, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['tileset_primary']['datatype'])
        assert(content['label'] == label_primary)
        self.tileset_primary = content['data']
        self.tileset_primary_label = label_primary
        properties.set_member_by_path(self.footer, label_primary, self.project.config['pymap']['footer']['tileset_primary_path'])
        path_secondary = self.project.tilesets_secondary[label_secondary]
        with open(path_secondary, encoding=self.project.config['json']['encoding']) as f:
            content = json.load(f)
        assert(content['type'] == self.project.config['pymap']['tileset_secondary']['datatype'])
        assert(content['label'] == label_secondary)
        self.tileset_secondary = content['data']
        self.tileset_secondary_label = label_secondary
        properties.set_member_by_path(self.footer, label_secondary, self.project.config['pymap']['footer']['tileset_secondary_path'])
        # Load the gfx and render tiles
        self.tiles = render.get_tiles(self.tileset_primary, self.tileset_secondary, self.project)
        self.blocks = render.get_blocks(self.tileset_primary, self.tileset_secondary, self.tiles, self.project)
        self._update()

    def _update(self):
        self.map_widget.load_header()
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

    def set_blocks(self, x, y, blocks, aggregate=False):
        """ Sets the blocks on the header and adds an item to the history. """
        if self.project is None or self.header is None: return
        map_blocks = properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])
        # Truncate blocks to fit the map
        window = map_blocks[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[:window.shape[0], :window.shape[1]].copy()
        self._do((HISTORY_SET_BLOCKS, (x, y, blocks, window)))

    def _do(self, action, aggregate=False):
        """ Executes an action and puts it as tail to the current history. """
        self.history = self.history[ : self.history_idx]
        self.history.append(action)
        self._redo()

    def _redo(self):
        """ Executes the last undone element in the history. """
        if self.history_idx in range(len(self.history)):
            action_type, parameters = self.history[self.history_idx]
            self.history_idx += 1
            if action_type == HISTORY_SET_BLOCKS:
                # Perform a set blocks action
                x, y, blocks_new, _ = parameters
                properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])[y : y + blocks_new.shape[0], x : x + blocks_new.shape[1]] = blocks_new
                self.map_widget.update_map(x, y, blocks_new)

    def _undo(self):
        """ Redos the last element in the history. """
        if self.history_idx > 0:
            self.history_idx -= 1
            action_type, parameters = self.history[self.history_idx]
            if action_type == HISTORY_SET_BLOCKS:
                # Undo a set blocks action
                x, y, _, blocks_old = parameters
                properties.get_member_by_path(self.footer, self.project.config['pymap']['footer']['map_blocks_path'])[y : y + blocks_old.shape[0], x : x + blocks_old.shape[1]] = blocks_old
                self.map_widget.update_map(x, y, blocks_old)
                




        
def main():
    #os.chdir('/media/d/romhacking/Violet_Sources')
    app = QApplication(sys.argv)
    ex = PymapGui()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()