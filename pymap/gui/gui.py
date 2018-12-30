# -*- coding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys, os
import appdirs
import resource_tree
import pymap.project
from settings import Settings

class PymapGui(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = Settings()
        self.project = None

        self.setup_ui()


    def setup_ui(self):
        # Build the menu bar
        self.file_menu = self.menuBar().addMenu('&File')
        self.file_menu_new_action = self.file_menu.addAction('&New')
        self.file_menu_new_action.setShortcut('Ctrl+N')
        self.file_menu_open_action = self.file_menu.addAction('&Open')
        self.file_menu_open_action.setShortcut('Ctrl+O')
        self.file_menu_open_action.triggered.connect(self.open_file)
        self.file_menu_save_action = self.file_menu.addAction('&Save')
        self.file_menu_save_action.setShortcut('Ctrl+S')
        self.edit_menu = self.menuBar().addMenu('&Edit')
        self.edit_menu_undo_action = self.edit_menu.addAction('Undo')
        self.edit_menu_undo_action.setShortcut('Ctrl+Z')
        self.edit_menu_redo_action = self.edit_menu.addAction('Redo')
        self.edit_menu_redo_action.setShortcut('Ctrl+Y')
        self.view_menu = self.menuBar().addMenu('&View')
        self.view_menu_resource_action = self.view_menu.addAction('Toggle Header Listing')
        self.view_menu_resource_action.setShortcut('Ctrl+L')
        self.view_menu_resource_action.triggered.connect(self.resource_tree_toggle_header_listing)

        
        
        # Add the project tree widget
        self.resource_tree_widget = QDockWidget('Project Resources')
        self.resource_tree = resource_tree.ResourceParameterTree()
        self.resource_tree_widget.setWidget(self.resource_tree)
        self.resource_tree_widget.setFloating(False)
        self.resource_tree_widget.setFeatures(QDockWidget.DockWidgetFloatable | 
                 QDockWidget.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.resource_tree_widget)

        # Add the tabs 
        self.central_widget = QTabWidget()
        self.map_widget = QTextEdit()
        self.level_widget = QTextEdit()
        self.header_widget = QTextEdit()
        self.footer_widget = QTextEdit()
        self.tileset_widget = QTextEdit()
        self.central_widget.addTab(self.map_widget, 'Map')
        self.central_widget.addTab(self.level_widget, 'Level')
        self.central_widget.addTab(self.header_widget, 'Header')
        self.central_widget.addTab(self.footer_widget, 'Footer')
        self.central_widget.addTab(self.tileset_widget, 'Tileset')

        self.setCentralWidget(self.central_widget)

    def open_file(self):
        """ Prompts a dialog to open a new project file. """
        path, suffix = QFileDialog.getOpenFileName(self, 'Open project', self.settings['recent.project'], 'Pymap projects (*.pmp)')
        if len(path):
            os.chdir(os.path.dirname(path))
            self.settings['recent.project'] = path
            self.project = pymap.project.Project(path)
            self.resource_tree.load_project(self.project, sort_headers = self.settings['resource_tree.header_listing'])
            self.update()

    def update(self):
        pass

    def resource_tree_toggle_header_listing(self):
        """ Toggles the listing method for the resource tree. """
        if self.project is None: return
        self.settings['resource_tree.header_listing'] = (
            resource_tree.SORT_BY_BANK if self.settings['resource_tree.header_listing'] == resource_tree.SORT_BY_NAMESPACE else
            resource_tree.SORT_BY_NAMESPACE
        )
        self.resource_tree.load_headers(self.project, sort_headers = self.settings['resource_tree.header_listing'])
        
def main():
    #os.chdir('/media/d/romhacking/Violet_Sources')
    app = QApplication(sys.argv)
    ex = PymapGui()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()