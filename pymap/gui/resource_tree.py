# Module for the resource tree widget
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os

SORT_BY_BANK = 'bank'
SORT_BY_NAMESPACE = 'namespace'

# Load icons
dir, _ = os.path.split(__file__)
icon_paths = {
    'header' : os.path.join(dir, 'icon', 'project_tree_header.png'),
    'folder' : os.path.join(dir, 'icon', 'project_tree_folder.png'),
    'tree' : os.path.join(dir, 'icon', 'project_tree_tree.png'),
    'footer' : os.path.join(dir, 'icon', 'project_tree_footer.png'),
    'tileset' : os.path.join(dir, 'icon', 'project_tree_tileset.png'),
    'gfx' : os.path.join(dir, 'icon', 'project_tree_gfx.png')
} 


class ResourceParameterTree(QTreeWidget):
    """ Widget for the resource tree. """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tree_header = QTreeWidgetItem(['Resource'])
        self.setHeaderItem(self.tree_header)
        self.header_root = QTreeWidgetItem(self, ['Map Header'])
        self.header_root.setIcon(0, QIcon(icon_paths['tree']))
        self.footer_root = QTreeWidgetItem(self, ['Map Footer'])
        self.footer_root.setIcon(0, QIcon(icon_paths['tree']))
        self.tileset_root = QTreeWidgetItem(self, ['Tileset'])
        self.tileset_root.setIcon(0, QIcon(icon_paths['tree']))
        self.gfx_root = QTreeWidgetItem(self, ['Gfx'])
        self.gfx_root.setIcon(0, QIcon(icon_paths['tree']))

    def load_project(self, project, sort_headers=SORT_BY_NAMESPACE):
        """ Displays the tree of a project.
        
        Parameters:
        -----------
        project : pymap.project.Project
            The underlying pymap project.
        sort_headers : str
            How the maps will be sorted.
        """
        self.load_headers(project, sort_headers=sort_headers)

        # Remove old footers
        remove_children(self.footer_root)
        # Add new footers
        for footer in sorted(project.footers, key=(lambda label: project.footers[label][0])):
            footer_idx, path = project.footers[footer]
            footer_root = QTreeWidgetItem(self.footer_root, [f'[{str(footer_idx).zfill(3)}] {footer}'])
            footer_root.setIcon(0, QIcon(icon_paths['footer']))

        # Remove old tilesets
        remove_children(self.tileset_root)
        # Add new tilesets
        for tileset in sorted(project.tilesets):
            tileset_root = QTreeWidgetItem(self.tileset_root, [f'{tileset}'])
            tileset_root.setIcon(0, QIcon(icon_paths['tileset']))

        # Remove old gfx
        remove_children(self.gfx_root)
        for gfx in sorted(project.gfxs):
            gfx_root = QTreeWidgetItem(self.gfx_root, [f'{gfx}'])
            gfx_root.setIcon(0, QIcon(icon_paths['gfx']))
    
    def load_headers(self, project, sort_headers=SORT_BY_NAMESPACE):
        """ Displays the headers of a project.
        
        Parameters:
        -----------
        project : pymap.project.Project
            The underlying pymap project.
        sort_headers : str
            How the maps will be sorted.
        """
        # Remove old headers
        remove_children(self.header_root)
        # Add new headers
        if sort_headers == SORT_BY_BANK:
            for bank in sorted(project.headers, key=int):
                bank_root = QTreeWidgetItem(self.header_root, [f'Bank {bank}'])
                bank_root.setIcon(0, QIcon(icon_paths['folder']))
                for map_idx in sorted(project.headers[bank], key=int):
                    label, path, namespace = project.headers[bank][map_idx]
                    map_root = QTreeWidgetItem(bank_root, [f'[{bank}, {map_idx.zfill(2)}] {label}'])
                    map_root.setIcon(0, QIcon(icon_paths['header']))
        elif sort_headers == SORT_BY_NAMESPACE:
            namespace_roots = {}
            for bank in sorted(project.headers, key=int):
                for map_idx in sorted(project.headers[bank], key=int):
                    label, path, namespace = project.headers[bank][map_idx]
                    if namespace not in namespace_roots:
                        namespace_roots[namespace] = QTreeWidgetItem(self.header_root, [str(namespace)])
                        namespace_roots[namespace].setIcon(0, QIcon(icon_paths['folder']))
                    map_root = QTreeWidgetItem(namespace_roots[namespace], [f'[{bank}, {map_idx.zfill(2)}] {label}'])
                    map_root.setIcon(0, QIcon(icon_paths['header']))

def remove_children(widget):
    """ Helper method to clear all children of a widget. """
    children = [widget.child(idx) for idx in range(widget.childCount())]
    for child in children:
        widget.removeChild(child)