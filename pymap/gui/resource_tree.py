# Module for the resource tree widget
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
from functools import partial
from properties import get_member_by_path, set_member_by_path 
import json

SORT_BY_BANK = 'bank'
SORT_BY_NAMESPACE = 'namespace'

HEADER_ROOT = 'header_root'
BANK = 'bank'
HEADER = 'header'
NAMESPACE = 'namespace'
FOOTER_ROOT = 'footer_root'
TILESET_PRIMARY = 'tileset_primary'
TILESET_SECONDARY = 'tileset_secondary'
FOOTER = 'footer'

# Load icons
dir, _ = os.path.split(__file__)
icon_paths = {
    'header' : os.path.join(dir, 'icon', 'project_tree_header.png'),
    'folder' : os.path.join(dir, 'icon', 'project_tree_folder.png'),
    'tree' : os.path.join(dir, 'icon', 'project_tree_tree.png'),
    'footer' : os.path.join(dir, 'icon', 'project_tree_footer.png'),
    'tileset' : os.path.join(dir, 'icon', 'project_tree_tileset.png'),
    'gfx' : os.path.join(dir, 'icon', 'project_tree_gfx.png'),
    'plus' : os.path.join(dir, 'icon', 'plus.png'),
    'remove' : os.path.join(dir, 'icon', 'remove.png')
} 


class ResourceParameterTree(QTreeWidget):
    """ Widget for the resource tree. """
    
    def __init__(self, main_gui, parent=None):
        super().__init__(parent)
        self.main_gui = main_gui
        self.tree_header = QTreeWidgetItem(['Resource'])
        self.setHeaderItem(self.tree_header)
        self.header_root = QTreeWidgetItem(self, ['Map Header'])
        self.header_root.setIcon(0, QIcon(icon_paths['tree']))
        self.header_root.context = HEADER_ROOT
        self.footer_root = QTreeWidgetItem(self, ['Map Footer'])
        self.footer_root.setIcon(0, QIcon(icon_paths['tree']))
        self.footer_root.context = FOOTER_ROOT
        self.tileset_root = QTreeWidgetItem(self, ['Tileset'])
        self.tileset_root.setIcon(0, QIcon(icon_paths['tree']))
        self.tileset_primary_root = QTreeWidgetItem(self.tileset_root, ['Primary'])
        self.tileset_primary_root.setIcon(0, QIcon(icon_paths['folder']))
        self.tileset_secondary_root = QTreeWidgetItem(self.tileset_root, ['Secondary'])
        self.tileset_secondary_root.setIcon(0, QIcon(icon_paths['folder']))
        self.gfx_root = QTreeWidgetItem(self, ['Gfx'])
        self.gfx_root.setIcon(0, QIcon(icon_paths['tree']))
        self.gfx_primary_root = QTreeWidgetItem(self.gfx_root, ['Primary'])
        self.gfx_primary_root.setIcon(0, QIcon(icon_paths['folder']))
        self.gfx_secondary_root = QTreeWidgetItem(self.gfx_root, ['Secondary'])
        self.gfx_secondary_root.setIcon(0, QIcon(icon_paths['folder']))

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

        self.itemDoubleClicked.connect(self.item_double_clicked)
    
    def context_menu_requested(self, position):
        """ Spawns a context menu at a given position if possible. """
        if self.main_gui.project is None: return
        item = self.itemAt(position)
        context = getattr(item, 'context', None)
        menu = QMenu()
        if context == HEADER_ROOT:
            if len(self.main_gui.project.unused_banks()) == 0:
                return # Spawn no context menu for the header root if there are no map banks to add
            action = menu.addAction('Add Bank')
            action.triggered.connect(self.create_bank)
            action.setIcon(QIcon(icon_paths['plus']))
        elif context == BANK:
            bank = item.context_data
            if len(self.main_gui.project.unused_map_idx(bank)) > 0:
                action = menu.addAction('Add Header')
                action.triggered.connect(partial(self.create_header, bank=bank))
                action.setIcon(QIcon(icon_paths['plus']))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_bank, bank=bank))
            action.setIcon(QIcon(icon_paths['remove']))
        elif context == HEADER:
            bank, map_idx = item.context_data
            action = menu.addAction('Open')
            action.triggered.connect(partial(self.main_gui.open_header, bank, map_idx))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_header, bank=bank, map_idx=map_idx))
            action.setIcon(QIcon(icon_paths['remove']))
        elif context == FOOTER_ROOT:
            action = menu.addAction('Add Footer')
            action.triggered.connect(self.create_footer)
            action.setIcon(QIcon(icon_paths['plus']))
        elif context == TILESET_PRIMARY:
            action = menu.addAction('Assign to Footer')
            action.triggered.connect(lambda _: self.main_gui.open_tilesets(label_primary=item.context_data))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_tileset, primary=True, label=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))
        elif context == TILESET_SECONDARY:
            action = menu.addAction('Assign to Footer')
            action.triggered.connect(lambda _: self.main_gui.open_tilesets(label_secondary=item.context_data))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_tileset, primary=False, label=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))
        elif context == FOOTER:
            action = menu.addAction('Assign to Header')
            action.triggered.connect(lambda _: self.main_gui.open_footer(item.context_data))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_footer, footer=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))

        else:
            return
        menu.exec_(self.viewport().mapToGlobal(position))

    def item_double_clicked(self, item, column_idx):
        """ Triggered when an item is double clicked. """
        context = getattr(item, 'context', None)
        if context == HEADER:
            self.main_gui.open_header(*(item.context_data))
        
            
    def create_bank(self, *args):
        """ Prompts a dialog to create a new map bank. """
        if self.main_gui.project is None: return
        unused_banks = self.main_gui.project.unused_banks()
        if len(unused_banks) == 0: return
        bank, ok_pressed = QInputDialog.getItem(self, 'Create New Map Bank', 'Enter the index of the map bank to create:', unused_banks, 0, False)
        if ok_pressed:
            self.main_gui.project.headers[bank] = {}
            self.load_headers()

    def remove_bank(self, bank=None):
        """ Removes a bank from the project with prompt. """
        pressed = QMessageBox.question(self, 'Confirm bank removal', f'Do you really want to remove bank {bank} from the project entirely?')
        if pressed == QMessageBox.Yes:
            del self.main_gui.project.headers[bank]
            self.load_headers()

    def create_header(self, *args, bank=None, map_idx=None, namespace=None, label=None, footer=None, path=None):
        """ Prompts a dialog to create a new header. """
        if self.main_gui.project is None: return
        # Check if there is space in any bank for a new header
        available_banks = sorted(list(filter(
                lambda _bank: len(self.main_gui.project.unused_map_idx(_bank)) > 0, 
                self.main_gui.project.headers.keys())
                ), key=lambda _bank: int(_bank))
        if len(available_banks) == 0:
            return QMessageBox.critical(self, 'Unable To Create New Header', 'There is no space for new headers in any map bank.')
        # Check if there are footers to associate the header with
        available_footers = list(sorted(self.main_gui.project.footers, key=(lambda footer_label: self.main_gui.project.footers[footer_label][0])))
        if len(available_footers) == 0:
            return QMessageBox.critical(self, 'Unable to Create New Header', 'There is no footer to associate the header with.')
        if bank is None:
            # Do not allow banks that are already full
            # Prompt the map bank
            bank, ok_pressed = QInputDialog.getItem(self, 'Create New Header', 'Select the bank to create the header in:', available_banks, 0, False)
            if not ok_pressed: return
        if map_idx is None:
            # Prompt the map idx
            map_idx, ok_pressed = QInputDialog.getItem(self, 'Create New Header', f'Select the index of the header in bank {bank}:', self.main_gui.project.unused_map_idx(bank), 0, False)
            if not ok_pressed: return
        if namespace is None:
            available_namespaces, namespaces_strict = self.main_gui.project.available_namespaces()
            # Prompt the namespace
            if len(available_namespaces) > 0:
                namespace, ok_pressed = QInputDialog.getItem(self, 'Create New Header', f'Select a namespace for the header:', available_namespaces, 0, namespaces_strict)
            else:
                namespace, ok_pressed = QInputDialog.getText(self, 'Create New Header', f'Enter a namespace for the header:')
            if not ok_pressed: return
        while label is None:
            # Prompt the label
            label, ok_pressed = QInputDialog.getText(self, 'Create New Header', f'Select a unique label for the header:')
            if not ok_pressed: return
            # Check if label already exists
            for bank in self.main_gui.project.headers:
                if label is None: break
                for map_idx in self.main_gui.project.headers[bank]:
                    if label is None: break
                    if label == self.project.headers[bank][map_idx][0]:
                        QMessageBox.critical(self, 'Inavlid label', f'Label {label} already exists at [{bank}, {map_idx.zfill(2)}')
                        label = None
        if footer is None:
            footer, ok_pressed = QInputDialog.getItem(self, 'Create New Header', 'Select a footer this map is associated with:', available_footers, 0, False)
            if not ok_pressed: return
            footer_idx = self.main_gui.project.footers[footer][0]
        if path is None:
            # Prompt the file path and create the new file
            path, suffix = QFileDialog.getSaveFileName(
                self, 'Create New Map Header', os.path.join(os.path.dirname(self.main_gui.settings['recent.header']), 
                f'{bank}_{map_idx}_{label}.pms'), 'Pymap Structure (*.pms)')
            if not len(path): return
            self.main_gui.settings['recent.header'] = path
            # Create new map file
            datatype = self.main_gui.project.config['pymap']['header']['datatype']
            header = self.main_gui.project.model[datatype](self.main_gui.project, [], [])
            # Assign the proper namespace
            set_member_by_path(header, footer, self.main_gui.project.config['pymap']['header']['footer_path'])
            set_member_by_path(header, footer_idx, self.main_gui.project.config['pymap']['header']['footer_idx_path'])
            set_member_by_path(header, namespace, self.main_gui.project.config['pymap']['header']['namespace_path'])
            with open(path, 'w+', encoding=self.main_gui.project.config['json']['encoding']) as f:
                json.dump({
                    'type' : self.main_gui.project.config['pymap']['header']['datatype'],
                    'label' : label,
                    'data' : header
                }, f, indent=self.main_gui.project.config['json']['indent'])
        # Add the map to the project structure
        self.main_gui.project.headers[bank][map_idx] = label, os.path.relpath(path), namespace
        self.load_headers()

    def remove_header(self, *args, bank=None, map_idx=None):
        label, path, namespace = self.main_gui.project.headers[bank][map_idx]
        pressed = QMessageBox.question(self, 'Confirm header removal', f'Do you really want to remove header [{bank}, {map_idx.zfill(2)}] {label} from the project entirely?')
        if pressed == QMessageBox.Yes:
            if label == self.main_gui.header_label:
                self.main_gui.clear_header()
            del self.main_gui.project.headers[bank][map_idx]
            self.load_headers()

    def create_footer(self, *args, footer_idx=None, tileset_primary=None, tileset_secondary=None, label=None, path=None):
        """ Prompts a dialog to create a new map footer. """
        # Check if there is space in the footer table
        available_idx = self.main_gui.project.unused_footer_idx()
        if not available_idx:
            return QMessageBox.critical(self, 'Unable To Create Footer', 'There are no available footer index. Remove another footer to add a new one first.')
        # Check if there are primary tilesets to use
        tilesets_primary = list(self.main_gui.project.tilesets_primary)
        if len(tilesets_primary) == 0:
            return QMessageBox.critical(self, 'Unable To Create Footer', 'There are no primary tilesets to use.', )
        # Check if there are secondary tilesets to use
        tilesets_secondary = list(self.main_gui.project.tilesets_secondary)
        if len(tilesets_secondary) == 0:
            return QMessageBox.critical(self, 'Unable To Create Footer', 'There are no secondary tilesets to use.', )
        if footer_idx is None:
            footer_idx, ok_pressed = QInputDialog.getItem(self, 'Create New Footer', 'Select the index in the footer table:', available_idx, 0, False)
            if not ok_pressed: return
        if tileset_primary is None:
            tileset_primary, ok_pressed = QInputDialog.getItem(self, 'Create New Footer', 'Select a primary tileset:', tilesets_primary, 0, False)
            if not ok_pressed: return
        if tileset_secondary is None:
            tileset_secondary, ok_pressed = QInputDialog.getItem(self, 'Create New Footer', 'Select a secondary tileset:', tilesets_secondary, 0, False)
            if not ok_pressed: return
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, 'Create New Footer', f'Select a unique label for the footer:')
            if not ok_pressed: return
            if label in self.main_gui.project.footers:
                QMessageBox.critical(self, 'Invalid label', f'The label {label} is already used for another footer.')
                label = None
        if path is None:
            # Prompt the file path and create the new file
            path, suffix = QFileDialog.getSaveFileName(
                self, 'Create New Footer', os.path.join(os.path.dirname(self.main_gui.settings['recent.footer']), 
                f'{label}.pms'), 'Pymap Structure (*.pms)')
            if not len(path): return
            self.main_gui.settings['recent.footer'] = path
            # Create new footer
            datatype = self.main_gui.project.config['pymap']['footer']['datatype']
            footer = self.main_gui.project.model[datatype](self.main_gui.project, [], [])
            # Assign the proper namespace
            set_member_by_path(footer, tileset_primary, self.main_gui.project.config['pymap']['footer']['tileset_primary_path'])
            set_member_by_path(footer, tileset_secondary, self.main_gui.project.config['pymap']['footer']['tileset_secondary_path'])
            # Set size measures to 1
            for member_path in ('map_width_path', 'map_height_path', 'border_width_path', 'border_height_path'):
                set_member_by_path(footer, 1, self.main_gui.project.config['pymap']['footer'][member_path])
            with open(path, 'w+', encoding=self.main_gui.project.config['json']['encoding']) as f:
                json.dump({
                    'type' : self.main_gui.project.config['pymap']['footer']['datatype'],
                    'label' : label,
                    'data' : footer
                }, f, indent=self.main_gui.project.config['json']['indent'])
        # Add the footer to the project structure
        self.main_gui.project.footers[label] = int(footer_idx), os.path.relpath(path)
        self.load_footers()

    def remove_footer(self, *args, footer=None):
        """ Removes a footer from the project with prompt. """
        pressed = QMessageBox.question(self, 'Confirm footer removal', f'Do you really want to remove footer {footer} from the project entirely?')
        if pressed == QMessageBox.Yes:
            # Scan through the entire project and change all references to this footer to None
            headers = []
            for bank in self.main_gui.project.headers:
                for map_idx in self.main_gui.project.headers[bank]:
                    label, path, namespace = self.main_gui.project.headers[bank][map_idx]
                    with open(path, encoding=self.main_gui.project.config['json']['encoding']) as f:
                        header = json.load(f)
                        if footer == get_member_by_path(header['data'], self.main_gui.project.config['pymap']['header']['footer_path']):
                            headers.append((bank, map_idx))
            if len(headers) > 0:
                headers_readable = [f'[{header[0]}, {header[1].zfill(2)}] {self.main_gui.project.headers[header[0]][header[1]][0]}' for header in headers]
                return QMessageBox.critical(self, 'Confirm footer removal', f'The following headers refer to footer {footer}: {", ".join(headers_readable)}. Assign different footers to those headers first.')
            del self.main_gui.project.footers[footer]
            self.load_footers()
        
    def create_tileset(self, *args, primary=None, gfx=None, label=None, path=None):
        """ Prompts a dialog to create a new tileset. """
        if primary is None:
            primary, ok_pressed = QInputDialog.getItem(self, 'Create New Tileset', 'Select the tileset type:', ['Primary', 'Secondary'], 0, False)
            if not ok_pressed: return
            primary = primary == 'Primary'
        gfxs = self.main_gui.project.gfxs_primary if primary else self.main_gui.project.gfxs_secondary
        tilesets = self.main_gui.project.tilesets_primary if primary else self.main_gui.project.tilesets_secondary
        tileset_type = 'tileset_primary' if primary else 'tileset_secondary'
        if len(gfxs) == 0:
            return QMessageBox.critical(self, 'Unable To Create Tileset', 'There are no gfxs avaialable for this tileset type.')
        if gfx is None:
            gfx, ok_pressed = QInputDialog.getItem(self, 'Create New Tileset', 'Select a gfx the tileset uses:', list(gfxs.keys()), 0, False)
            if not ok_pressed: return
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, 'Create New Tileset', f'Select a unique label for the tileset:')
            if not ok_pressed: return
            if label in tilesets:
                QMessageBox.critical(self, 'Invalid label', f'The label {label} is already used for another tileset of this type.')
                label = None
        if path is None:
            # Prompt the file path and create the new file
            path, suffix = QFileDialog.getSaveFileName(
                self, 'Create New Tileset', os.path.join(os.path.dirname(self.main_gui.settings['recent.tileset']), 
                f'{label}.pms'), 'Pymap Structure (*.pms)')
            if not len(path): return
            self.main_gui.settings['recent.tileset'] = path
            # Create new footer
            datatype = self.main_gui.project.config['pymap'][tileset_type]['datatype']
            tileset = self.main_gui.project.model[datatype](self.main_gui.project, [], [])
            # Assign the proper namespace
            set_member_by_path(tileset, gfx, self.main_gui.project.config['pymap'][tileset_type]['gfx_path'])
            with open(path, 'w+', encoding=self.main_gui.project.config['json']['encoding']) as f:
                json.dump({
                    'type' : self.main_gui.project.config['pymap'][tileset_type]['datatype'],
                    'label' : label,
                    'data' : tileset
                }, f, indent=self.main_gui.project.config['json']['indent'])
        # Add the footer to the project structure
        tilesets[label] = oath
        self.load_tilesets()
        
    def remove_tileset(self, *args, primary=None, label=None):
        """ Removes a primary tileset. """
        pressed = QMessageBox.question(self, 'Confirm tileset removal', f'Do you really want to remove tileset {label} from the project entirely?')
        if pressed == QMessageBox.Yes:
            # Scan through all footers and collect footers that refer to this primary / secondary tileset
            footers = []
            for footer_label in self.main_gui.project.footers:
                footer_idx, path = self.main_gui.project.footers[footer_label]
                with open(path, encoding=self.main_gui.project.config['json']['encoding']) as f:
                    footer = json.load(f)
                if label == get_member_by_path(footer['data'], self.main_gui.project.config['pymap']['footer']['tileset_primary_path' if primary else 'tileset_secondary_path']):
                    footers.append(footer_label)
            if len(footers) > 0:
                return QMessageBox.critical(self, 'Tileset Removal', f'The following footers refer to the tileset {label}: {", ".join(footers)}. Assign different tilesets to those footers first.')
            if primary:
                del self.main_gui.project.tilesets_primary[label]
            else:
                del self.main_gui.project.tilesets_secondary[label]
            self.load_footers()

    def load_project(self):
        """ Updates the tree of a project. """
        self.load_headers()
        self.load_footers()
        self.load_tilesets()
        self.load_gfx()
    
    def load_headers(self):
        """ Updates the headers of a project. """
        project = self.main_gui.project
        sort_headers = self.main_gui.settings['resource_tree.header_listing']
        # Remove old headers
        remove_children(self.header_root)
        # Add new headers
        if sort_headers == SORT_BY_BANK:
            for bank in sorted(project.headers, key=int):
                bank_root = QTreeWidgetItem(self.header_root, [f'Bank {bank}'])
                bank_root.setIcon(0, QIcon(icon_paths['folder']))
                bank_root.context, bank_root.context_data = BANK, bank
                for map_idx in sorted(project.headers[bank], key=int):
                    label, path, namespace = project.headers[bank][map_idx]
                    map_root = QTreeWidgetItem(bank_root, [f'[{bank}, {map_idx.zfill(2)}] {label}'])
                    map_root.setIcon(0, QIcon(icon_paths['header']))
                    map_root.context, map_root.context_data = HEADER, (bank, map_idx)
        elif sort_headers == SORT_BY_NAMESPACE:
            namespace_roots = {}
            for bank in sorted(project.headers, key=int):
                for map_idx in sorted(project.headers[bank], key=int):
                    label, path, namespace = project.headers[bank][map_idx]
                    if namespace not in namespace_roots:
                        namespace_roots[namespace] = QTreeWidgetItem(self.header_root, [str(namespace)])
                        namespace_roots[namespace].setIcon(0, QIcon(icon_paths['folder']))
                        namespace_roots[namespace].context = NAMESPACE
                    map_root = QTreeWidgetItem(namespace_roots[namespace], [f'[{bank}, {map_idx.zfill(2)}] {label}'])
                    map_root.setIcon(0, QIcon(icon_paths['header']))
                    map_root.context, map_root.context_data = HEADER, (bank, map_idx)

    def load_footers(self):
        """ Updates the footers of a project. """
        project = self.main_gui.project
        # Remove old footers
        remove_children(self.footer_root)
        # Add new footers
        for footer in sorted(project.footers, key=(lambda label: project.footers[label][0])):
            footer_idx, path = project.footers[footer]
            footer_root = QTreeWidgetItem(self.footer_root, [f'[{str(footer_idx).zfill(3)}] {footer}'])
            footer_root.setIcon(0, QIcon(icon_paths['footer']))
            footer_root.context, footer_root.context_data = FOOTER, footer

    def load_tilesets(self):
        """ Updates the tilesets of a project. """
        project = self.main_gui.project
        # Remove old tilesets
        remove_children(self.tileset_primary_root)
        remove_children(self.tileset_secondary_root)
        # Add new tilesets
        for tileset in sorted(project.tilesets_primary):
            tileset_root = QTreeWidgetItem(self.tileset_primary_root, [f'{tileset}'])
            tileset_root.setIcon(0, QIcon(icon_paths['tileset']))
            tileset_root.context, tileset_root.context_data = TILESET_PRIMARY, tileset
        for tileset in sorted(project.tilesets_secondary):
            tileset_root = QTreeWidgetItem(self.tileset_secondary_root, [f'{tileset}'])
            tileset_root.setIcon(0, QIcon(icon_paths['tileset']))
            tileset_root.context, tileset_root.context_data = TILESET_SECONDARY, tileset

    def load_gfx(self):
        """ Updates the gfxs of a project. """
        project = self.main_gui.project
        # Remove old gfx
        remove_children(self.gfx_primary_root)
        remove_children(self.gfx_secondary_root)
        for gfx in sorted(project.gfxs_primary):
            gfx_root = QTreeWidgetItem(self.gfx_primary_root, [f'{gfx}'])
            gfx_root.setIcon(0, QIcon(icon_paths['gfx']))
        for gfx in sorted(project.gfxs_secondary):
            gfx_root = QTreeWidgetItem(self.gfx_secondary_root, [f'{gfx}'])
            gfx_root.setIcon(0, QIcon(icon_paths['gfx']))


def remove_children(widget):
    """ Helper method to clear all children of a widget. """
    children = [widget.child(idx) for idx in range(widget.childCount())]
    for child in children:
        widget.removeChild(child)


