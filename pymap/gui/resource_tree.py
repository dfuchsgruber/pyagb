# Module for the resource tree widget
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
from functools import partial
from .properties import get_member_by_path, set_member_by_path 
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
TILESET_PRIMARY_ROOT = 'tileset_primary_root'
TILESET_SECONDARY_ROOT = 'tileset_secondary_root'
GFX_PRIMARY_ROOT = 'gfx_primary_root'
GFX_SECONDARY_ROOT = 'gfx_secondary_root'
GFX_PRIMARY = 'gfx_primary'
GFX_SECONDARY = 'gfx_secondary'
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
    'remove' : os.path.join(dir, 'icon', 'remove.png'),
    'import' : os.path.join(dir, 'icon', 'import.png'),
    'rename' : os.path.join(dir, 'icon', 'rename.png'),
    'tag' : os.path.join(dir, 'icon', 'tag.png')
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
        self.tileset_primary_root.context = TILESET_PRIMARY_ROOT
        self.tileset_secondary_root = QTreeWidgetItem(self.tileset_root, ['Secondary'])
        self.tileset_secondary_root.setIcon(0, QIcon(icon_paths['folder']))
        self.tileset_secondary_root.context = TILESET_SECONDARY_ROOT
        self.gfx_root = QTreeWidgetItem(self, ['Gfx'])
        self.gfx_root.setIcon(0, QIcon(icon_paths['tree']))
        self.gfx_primary_root = QTreeWidgetItem(self.gfx_root, ['Primary'])
        self.gfx_primary_root.context = GFX_PRIMARY_ROOT
        self.gfx_primary_root.setIcon(0, QIcon(icon_paths['folder']))
        self.gfx_secondary_root = QTreeWidgetItem(self.gfx_root, ['Secondary'])
        self.gfx_secondary_root.context = GFX_SECONDARY_ROOT
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
            action = menu.addAction('Import Header')
            action.setIcon(QIcon(icon_paths['import']))
            action.triggered.connect(lambda _: self.import_header())
        elif context == BANK:
            bank = item.context_data
            if len(self.main_gui.project.unused_map_idx(bank)) > 0:
                action = menu.addAction('Add Header')
                action.triggered.connect(partial(self.create_header, bank=bank))
                action.setIcon(QIcon(icon_paths['plus']))
                action = menu.addAction('Import Header')
                action.setIcon(QIcon(icon_paths['import']))
                action.triggered.connect(lambda _: self.import_header(bank=bank))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_bank, bank=bank))
            action.setIcon(QIcon(icon_paths['remove']))
        elif context == HEADER:
            bank, map_idx = item.context_data
            label, _, namespace = self.main_gui.project.headers[bank][map_idx]
            action = menu.addAction('Open')
            action.triggered.connect(partial(self.main_gui.open_header, bank, map_idx))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_header, bank=bank, map_idx=map_idx))
            action.setIcon(QIcon(icon_paths['remove']))
            action = menu.addAction('Relabel')
            action.setIcon(QIcon(icon_paths['rename']))
            action.triggered.connect(lambda _: self.refactor_header(bank=bank, map_idx=map_idx, namespace=namespace))
            action = menu.addAction('Change Namespace')
            action.setIcon(QIcon(icon_paths['tag']))
            action.triggered.connect(lambda _: self.refactor_header(bank=bank, map_idx=map_idx, label=label))
        elif context == FOOTER_ROOT:
            action = menu.addAction('Add Footer')
            action.triggered.connect(self.create_footer)
            action.setIcon(QIcon(icon_paths['plus']))
            action = menu.addAction('Import Footer')
            action.setIcon(QIcon(icon_paths['import']))
            action.triggered.connect(lambda _: self.import_footer())
        elif context == TILESET_PRIMARY or context == TILESET_SECONDARY:
            action = menu.addAction('Assign to Footer')
            action.triggered.connect(lambda _: self.main_gui.change_tileset(item.context_data, context == TILESET_SECONDARY))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_tileset, primary=context == TILESET_SECONDARY, label=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))
            action = menu.addAction('Relabel')
            action.setIcon(QIcon(icon_paths['rename']))
            action.triggered.connect(lambda _: self.refactor_tileset(primary=context == TILESET_SECONDARY, label_old=item.context_data))
        elif context == FOOTER:
            action = menu.addAction('Assign to Header')
            action.triggered.connect(lambda _: self.main_gui.change_footer(item.context_data))
            action = menu.addAction('Remove')
            action.triggered.connect(partial(self.remove_footer, footer=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))
            action = menu.addAction('Relabel')
            action.setIcon(QIcon(icon_paths['rename']))
            action.triggered.connect(lambda _: self.refactor_footer(label_old=item.context_data))
        elif context == NAMESPACE:
            action = menu.addAction('Add Header')
            action.setIcon(QIcon(icon_paths['plus']))
            action.triggered.connect(partial(self.create_header, namespace=str(item.context_data)))
        elif context == TILESET_PRIMARY_ROOT or context == TILESET_SECONDARY_ROOT:
            action = menu.addAction('Add Tileset')
            action.triggered.connect(lambda _: self.create_tileset(primary=context==TILESET_PRIMARY_ROOT))
            action.setIcon(QIcon(icon_paths['plus']))
            action = menu.addAction('Import Tileset')
            action.triggered.connect(lambda _: self.import_tileset(primary=context==TILESET_PRIMARY_ROOT))
            action.setIcon(QIcon(icon_paths['import']))
        elif context == GFX_PRIMARY or context == GFX_SECONDARY:
            action = menu.addAction('Assign to Tileset')
            action.triggered.connect(lambda _: self.main_gui.change_gfx(item.context_data, context == GFX_PRIMARY))
            action = menu.addAction('Remove')
            action.triggered.connect(lambda _: self.remove_gfx(primary=context == GFX_PRIMARY, label=item.context_data))
            action.setIcon(QIcon(icon_paths['remove']))
            action = menu.addAction('Relabel')
            action.triggered.connect(lambda _: self.refactor_gfx(primary=context == GFX_PRIMARY, label_old=item.context_data))
            action.setIcon(QIcon(icon_paths['rename']))
        elif context == GFX_PRIMARY_ROOT or context == GFX_SECONDARY_ROOT:
            action = menu.addAction('Import Gfx')
            action.setIcon(QIcon(icon_paths['import']))
            action.triggered.connect(lambda _: self.import_gfx(primary=context == GFX_PRIMARY_ROOT))
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
            for map_idx in self.main_gui.project.headers[bank]:
                if bank == self.main_gui.header_bank and map_idx == self.main_gui.header_map_idx:
                    self.main_gui.clear_header()
            self.main_gui.project.remove_bank(bank)
            self.load_headers()

    def refactor_header(self, *args, bank=None, map_idx=None, label=None, namespace=None):
        """ Refactors a map header. """
        if self.main_gui.project is None: return#
        if label is None:
            label = self.prompt_header_label(f'Refactor Header [{bank}, {map_idx.zfill(2)}]')
            if label is None: return
        if namespace is None:
            namespace = self.prompt_namespace(f'Refactor Header [{bank}, {map_idx.zfill(2)}]')
            if namespace is None: return
        self.main_gui.project.refactor_header(bank, map_idx, label, namespace)
        self.load_headers()
        self.main_gui.update()
    
    def prompt_header_label(self, title):
        """ Prompts a label for a header. """
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, title, f'Select a unique label for the header:')
            if not ok_pressed: return None
            # Check if label already exists
            for other_bank in self.main_gui.project.headers:
                if label is None: break
                for other_map_idx in self.main_gui.project.headers[other_bank]:
                    if label is None: break
                    if label == self.main_gui.project.headers[other_bank][other_map_idx][0]:
                        QMessageBox.critical(self, 'Inavlid label', f'Label {label} already exists at [{other_bank}, {other_map_idx.zfill(2)}]')
                        label = None
            return label
        
    def prompt_namespace(self, title):
        """ Prompts a namespace"""
        available_namespaces, namespaces_strict = self.main_gui.project.available_namespaces()
        print(namespaces_strict)
        # Prompt the namespace
        if len(available_namespaces) > 0:
            namespace, ok_pressed = QInputDialog.getItem(self, title, f'Select a namespace for the header:', available_namespaces, 0, namespaces_strict)
        else:
            namespace, ok_pressed = QInputDialog.getText(self, title, f'Enter a namespace for the header:')
        if not ok_pressed: return None
        else: return namespace

    def prompt_non_full_bank(self, title):
        """ Prompts a map bank that is not full yet. """
        available_banks = sorted(list(filter(
                lambda _bank: len(self.main_gui.project.unused_map_idx(_bank)) > 0, 
                self.main_gui.project.headers.keys())
                ), key=lambda _bank: int(_bank))
        if len(available_banks) == 0:
            QMessageBox.critical(self, 'All Banks Full', 'There is no space for new headers in any map bank.')
            return None
        bank, ok_pressed = QInputDialog.getItem(self, title, 'Select the bank to create the header in:', available_banks, 0, False)
        if not ok_pressed: return None
        else: return bank

    def prompt_unused_map_idx(self, title, bank):
        """ Prompts an unused map idx in a given bank. """
        unused_map_idx = self.main_gui.project.unused_map_idx(bank)
        if len(unused_map_idx) == 0:
            QMessageBox.critical(self, 'Bank Full', f'There is no space for new headers in bank {bank}.')
            return None
        map_idx, ok_pressed = QInputDialog.getItem(self, title, f'Select the index of the header in bank {bank}:', unused_map_idx, 0, False)
        if not ok_pressed: return None
        else: return map_idx

    def prompt_footer(self, title):
        """ Prompts a footer. """
        available_footers = list(sorted(self.main_gui.project.footers, key=(lambda footer_label: self.main_gui.project.footers[footer_label][0])))
        if len(available_footers) == 0:
            QMessageBox.critical(self, 'No Footers', 'There is no footer in the project to associate the header with.')
            return None
        footer, ok_pressed = QInputDialog.getItem(self, title, 'Select a footer this map is associated with:', available_footers, 0, False)
        if not ok_pressed: return None
        else: return footer

    def create_header(self, *args, bank=None, namespace=None):
        """ Prompts a dialog to create a new header. """
        if self.main_gui.project is None: return
        if bank is None: 
            bank = self.prompt_non_full_bank('Create New Header')
            if bank is None: return
        map_idx = self.prompt_unused_map_idx('Create New Header', bank)
        if map_idx is None: return
        footer = self.prompt_footer('Create Header')
        if footer is None: return
        if namespace is None:
            namespace = self.prompt_namespace('Create New Header')
            if namespace is None: return
        if label is None:
            label = self.prompt_header_label('Create New Header')
            if label is None: return
        footer_idx = self.main_gui.project.footers[footer][0]
        # Prompt the file path and create the new file
        path, suffix = QFileDialog.getSaveFileName(
            self, 'Create New Map Header', os.path.join(os.path.dirname(self.main_gui.settings['recent.header']), 
            f'{bank}_{map_idx}_{label}.pms'), 'Pymap Structure (*.pms)')
        if not len(path): return
        self.main_gui.settings['recent.header'] = path

        # Create new map file
        header = self.main_gui.project.new_header(label, path, namespace, bank, map_idx)
        # Assign the proper namespace, footer and footer index
        set_member_by_path(header, footer, self.main_gui.project.config['pymap']['header']['footer_path'])
        set_member_by_path(header, footer_idx, self.main_gui.project.config['pymap']['header']['footer_idx_path'])
        set_member_by_path(header, namespace, self.main_gui.project.config['pymap']['header']['namespace_path'])
        self.main_gui.project.save_header(header, bank, map_idx)
        self.load_headers()
        self.main_gui.update()

    def remove_header(self, *args, bank=None, map_idx=None):
        """ Removes a header from the project. """
        pressed = QMessageBox.question(self, 'Confirm header removal', f'Do you really want to remove header [{bank}, {map_idx.zfill(2)}] from the project entirely?')
        if pressed == QMessageBox.Yes:
            if bank == self.main_gui.header_bank and map_idx == self.main_gui.header_map_idx:
                self.main_gui.clear_header()
            self.main_gui.project.remove_header(bank, map_idx)
            self.load_headers()
            self.main_gui.update()

    def import_header(self, *args, bank=None, map_idx=None):
        """ Imports a map header structure into the project. """
        if bank is None:
            bank = self.prompt_non_full_bank('Import Header')
            if bank is None: return
        if map_idx is None:
            map_idx = self.prompt_unused_map_idx('Import Header', bank)
            if map_idx is None: return
        path, suffix = QFileDialog.getOpenFileName(
            self, 'Import Header', os.path.join(os.path.dirname(self.main_gui.settings['recent.header']), 
            f'{bank}_{map_idx}.pms'), 'Pymap Structure (*.pms)')
        self.main_gui.settings['recent.header'] = path
        if not len(path): return
        label = self.prompt_header_label('Import Header')
        if label is None: return
        namespace = self.prompt_namespace('Import Header')
        if namespace is None: return
        footer = self.prompt_footer('Import Footer')
        if footer is None: return
        self.main_gui.project.import_header(bank, map_idx, label, path, namespace, footer)
        self.load_headers()
        self.main_gui.update()
        
    def prompt_unused_footer_idx(self, title):
        """ Prompts a dialog that asks for an unused footer index. """
        available_idx = list(map(str, sorted(list(self.main_gui.project.unused_footer_idx()))))
        if not available_idx:
            QMessageBox.critical(self, title, 'There are no available footer index. Remove another footer to add a new one first.')
            return None
        footer_idx, ok_pressed = QInputDialog.getItem(self, title, 'Select the index in the footer table:', available_idx, 0, False)
        if not ok_pressed: return None
        else: return footer_idx

    def prompt_tileset(self, title, primary):
        """ Prompts a dialog that asks for a tileset. """
        tilesets = list(self.main_gui.project.tilesets_primary) if primary else list(self.main_gui.project.tilesets_secondary)
        if len(tilesets) == 0:
            return QMessageBox.critical(self, title, f'There are no {"primary" if primary else "secondary"} tilesets to assign to the footer.', )
        tileset, ok_pressed = QInputDialog.getItem(self, title, f'Select a {"primary" if primary else "secondary"} tileset', tilesets, 0, False)
        if not ok_pressed: return None
        else: return tileset

    def prompt_footer_label(self, title):
        """ Prompts a dialog to enter a unique label for a footer. """
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, title, f'Select a unique label for the footer:')
            if not ok_pressed: return None
            if label in self.main_gui.project.footers:
                QMessageBox.critical(self, 'Invalid label', f'The label {label} is already used for another footer.')
                label = None
        return label

    def create_footer(self, *args):
        """ Prompts a dialog to create a new map footer. """
        if self.main_gui.project is None: return
        footer_idx = self.prompt_unused_footer_idx('Create New Footer')
        if footer_idx is None: return
        tileset_primary = self.prompt_tileset('Create New Footer', True)
        if tileset_primary is None: return
        tileset_secondary = self.prompt_tileset('Create New Footer', False)
        if tileset_secondary is None: return
        label = self.prompt_footer_label('Create New Footer')
        if label is None: return

        path, suffix = QFileDialog.getSaveFileName(
            self, 'Create New Footer', os.path.join(os.path.dirname(self.main_gui.settings['recent.footer']), 
            f'{label}.pms'), 'Pymap Structure (*.pms)')
        if not len(path): return
        self.main_gui.settings['recent.footer'] = path

        # Create new footer
        footer = self.main_gui.project.new_footer(label, path, int(footer_idx))
        # Assign the tilesets
        set_member_by_path(footer, tileset_primary, self.main_gui.project.config['pymap']['footer']['tileset_primary_path'])
        set_member_by_path(footer, tileset_secondary, self.main_gui.project.config['pymap']['footer']['tileset_secondary_path'])
        self.main_gui.project.save_footer(footer, label)
        self.load_footers()
        self.main_gui.update()

    def remove_footer(self, *args, footer=None):
        """ Removes a footer from the project with prompt. """
        pressed = QMessageBox.question(self, 'Confirm footer removal', f'Do you really want to remove footer {footer} from the project entirely?')
        if pressed == QMessageBox.Yes:
            # Scan through the entire project and change all references to this footer to None
            headers = []
            for bank in self.main_gui.project.headers:
                for map_idx in self.main_gui.project.headers[bank]:
                    if map_idx == self.main_gui.header_map_idx and bank == self.main_gui.header_bank:
                        if self.main_gui.footer_label == footer:
                            headers.append((bank, map_idx))
                    else:
                        header, _, _ = self.main_gui.project.load_header(bank, map_idx)
                        if footer == get_member_by_path(header, self.main_gui.project.config['pymap']['header']['footer_path']):
                            headers.append((bank, map_idx))
            if len(headers) > 0:
                headers_readable = [f'[{header[0]}, {header[1].zfill(2)}] {self.main_gui.project.headers[header[0]][header[1]][0]}' for header in headers]
                return QMessageBox.critical(self, 'Confirm footer removal', f'The following headers refer to footer {footer}: {", ".join(headers_readable)}. Assign different footers to those headers first.')
            self.main_gui.project.remove_footer(footer)
            self.load_footers()
            self.main_gui.update()

    def refactor_footer(self, *args, label_old=None, label_new=None):
        """ Refactors the map footer's label. """
        if self.main_gui.project is None: return
        if label_new is None:
            label_new = self.prompt_footer_label('Relabel Map Footer')
            if label_new is None: return
        self.main_gui.project.refactor_footer(label_old, label_new)
        # If the current map refers to this footer, change the current label as well
        if self.main_gui.footer_label == label_old:
            self.main_gui.footer_label = label_new
            set_member_by_path(self.main_gui.header, label_new, self.main_gui.project.config['pymap']['header']['footer_path'])
        self.load_footers()
        self.main_gui.update()

    def import_footer(self, *args):
        """ Imports a map header structure into the project. """
        path, suffix = QFileDialog.getOpenFileName(
            self, 'Import Footer', os.path.join(os.path.dirname(self.main_gui.settings['recent.footer']), 
            f'footer.pms'), 'Pymap Structure (*.pms)')
        self.main_gui.settings['recent.footer'] = path
        if not len(path): return
        footer_idx = self.prompt_unused_footer_idx('Import Footer')
        if footer_idx is None: return
        label = self.prompt_footer_label('Import Footer')
        if label is None: return
        self.main_gui.project.import_footer(label, path, int(footer_idx))
        self.load_footers()
        self.main_gui.update()
        
    def prompt_gfx(self, title, primary):
        """ Prompts for a gfx by a dialog. """
        gfxs = self.main_gui.project.gfxs_primary if primary else self.main_gui.project.gfxs_secondary
        if len(gfxs) == 0:
            QMessageBox.critical(self, title, 'There are no gfxs avaialable for this tileset type.')
            return None
        gfx, ok_pressed = QInputDialog.getItem(self, title, 'Select a gfx the tileset uses:', list(gfxs.keys()), 0, False)
        if not ok_pressed: return None
        return gfx

    def prompt_tileset_label(self, title, primary):
        """ Prompts for a label of a tileset. """
        tilesets = self.main_gui.project.tilesets_primary if primary else self.main_gui.project.tilesets_secondary
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, title, f'Select a unique label for the tileset:')
            if not ok_pressed: return None
            if label in tilesets:
                QMessageBox.critical(self, title, f'The label {label} is already used for another tileset of this type.')
                label = None
        return label
    
    def prompt_gfx_label(self, title, primary):
        """ Prompts for an unused gfx label """
        gfxs = self.main_gui.project.gfxs_primary if primary else self.main_gui.project.gfxs_secondary
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(self, title, f'Select a unique label for the gfx:')
            if not ok_pressed: return None
            if label in gfxs:
                QMessageBox.critical(self, title, f'The label {label} is already used for another gfx of this type.')
                label = None
        return label

    def create_tileset(self, *args, primary=None):
        """ Prompts a dialog to create a new tileset. """
        if self.main_gui.project is None: return
        # Prompt the type if unspecified
        if primary is None:
            primary = self.prompt_tileset_type()
            if primary is None: return None
        gfx = self.prompt_gfx('Create Tileset', primary)
        if gfx is None: return
        label = self.prompt_tileset_label('Create Tileset', primary)
        if label is None: return
        path, suffix = QFileDialog.getSaveFileName(
            self, 'Create Tileset', os.path.join(os.path.dirname(self.main_gui.settings['recent.tileset']), 
            f'{label}.pms'), 'Pymap Structure (*.pms)')
        if not len(path): return
        self.main_gui.settings['recent.tileset'] = path
            
        # Create new tileset and assign the gfx
        tileset = self.main_gui.project.new_tileset(primary, label, path)
        set_member_by_path(tileset, gfx, self.main_gui.project.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['gfx_path'])
        self.main_gui.project.save_tileset(primary, tileset, label)
        self.load_tilesets()
        self.main_gui.update()
        
    def remove_tileset(self, *args, primary=None, label=None):
        """ Removes a tileset. """
        pressed = QMessageBox.question(self, 'Confirm tileset removal', f'Do you really want to remove tileset {label} from the project entirely?')
        if pressed == QMessageBox.Yes:
            # Scan through all footers and collect footers that refer to this primary / secondary tileset
            footers = []
            for footer_label in self.main_gui.project.footers:
                if footer_label == self.main_gui.footer_label:
                    if (label == self.main_gui.tileset_primary_label and primary) or (label == self.main_gui.tileset_secondary_label and not primary):
                        footers.append(footer_label)
                else:
                    footer, footer_idx = self.main_gui.project.load_footer(footer_label)
                    if label == get_member_by_path(footer, self.main_gui.project.config['pymap']['footer']['tileset_primary_path' if primary else 'tileset_secondary_path']):
                        footers.append(footer_label)
            if len(footers) > 0:
                return QMessageBox.critical(self, 'Tileset Removal', f'The following footers refer to the tileset {label}: {", ".join(footers)}. Assign different tilesets to those footers first.')
            self.main_gui.project.remove_tileset(primary, label)
            self.load_tilesets()
            self.main_gui.update()

    def refactor_tileset(self, *args, primary=None, label_old=None, label_new=None):
        """ Changes the label of a tileset. """
        if self.main_gui.project is None: return
        if label_new is None:
            label_new = self.prompt_tileset_label('Relabel Tileset', primary)
            if label_new is None: return
        self.main_gui.project.refactor_tileset(primary, label_old, label_new)
        # If the current footer refers to this tileset, change the label as well
        if primary and self.main_gui.tileset_primary_label == label_old:
            self.main_gui.tileset_primary_label = label_new
            set_member_by_path(self.main_gui.footer, label_new, self.main_gui.project.config['pymap']['footer']['tileset_primary_path'])
        elif not primary and self.main_gui.tileset_secondary_label == label_old:
            self.main_gui.tileset_secondary_label = label_new
            set_member_by_path(self.main_gui.footer, label_new, self.main_gui.project.config['pymap']['footer']['tileset_secondary_path'])
        self.load_tilesets()
        self.main_gui.update()

    def import_tileset(self, *args, primary=None):
        """ Imports a tileset. """
        if self.main_gui.project is None: return
        if primary is None:
            primary = self.prompt_tileset_type()
            if primary is None: return
        path, suffix = QFileDialog.getOpenFileName(
            self, 'Import Tileset', os.path.join(os.path.dirname(self.main_gui.settings['recent.tileset']), 
            f'tileset.pms'), 'Pymap Structure (*.pms)')
        self.main_gui.settings['recent.tileset'] = path
        if not len(path): return
        label = self.prompt_tileset_label('Import Tileset', primary)
        if label is None: return
        self.main_gui.project.import_tileset(primary, label, path)
        self.load_tilesets()
        self.main_gui.update()

    def refactor_gfx(self, *args, primary=None, label_old=None, label_new=None):
        """ Changes the label of a gfx. """
        if self.main_gui.project is None: return
        if label_new is None:
            label_new = self.prompt_gfx_label('Relabel Gfx', primary)
            if label_new is None: return
        self.main_gui.project.refactor_gfx(primary, label_old, label_new)
        # If the current tileset refers to this gfx, change the label as well
        if primary:
            gfx_primary_label = get_member_by_path(self.main_gui.tileset_primary, self.main_gui.project.config['pymap']['tileset_primary']['gfx_path'])
            if gfx_primary_label == label_old:
                set_member_by_path(self.main_gui.tileset_primary, label_new, self.main_gui.project.config['pymap']['tileset_primary']['gfx_path'])
        else:
            gfx_secondary_label = get_member_by_path(self.main_gui.tileset_secondary, self.main_gui.project.config['pymap']['tileset_secondary']['gfx_path'])
            if gfx_secondary_label == label_old:
                set_member_by_path(self.main_gui.tileset_secondary, label_new, self.main_gui.project.config['pymap']['tileset_secondary']['gfx_path'])
        self.load_gfx()
        self.main_gui.update()

    def remove_gfx(self, *args, primary=None, label=None):
        """ Removes a gfxs. """
        pressed = QMessageBox.question(self, 'Confirm gfx removal', f'Do you really want to remove gfx {label} from the project entirely?')
        if pressed == QMessageBox.Yes:
            # Scan through all tilesets and collect tilesets that refer to this gfx
            tilesets = []
            current_tileset = self.main_gui.tileset_primary_label if primary else self.main_gui.tileset_secondary_label
            for tileset_label in (self.main_gui.project.tilesets_primary if primary else self.main_gui.project.tilesets_secondary):
                # Check if currently the active tileset in display is refering to this gfx
                if (self.main_gui.tileset_primary_label == tileset_label and primary) or (self.main_gui.tileset_secondary_label == tileset_label and not primary):
                    tileset = self.main_gui.tileset_primary if primary else self.main_gui.tileset_secondary
                else:
                    tileset = self.main_gui.project.load_tileset(primary, tileset_label)
                if label == get_member_by_path(tileset, self.main_gui.project.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['gfx_path']):
                    tilesets.append(tileset_label)
            if len(tilesets) > 0:
                return QMessageBox.critical(self, 'Gfx Removal', f'The following tilesets refer to the gfx {label}: {", ".join(tilesets)}. Assign different gfxs to those tilesets first.')
            self.main_gui.project.remove_gfx(primary, label)
            self.load_gfx()
            self.main_gui.update()

    def import_gfx(self, *args, primary):
        """ Imports a gfx. """
        if self.main_gui.project is None: return
        path, suffix = QFileDialog.getOpenFileName(
            self, 'Import Gfx', os.path.join(os.path.dirname(self.main_gui.settings['recent.gfx']), 
            f'tileset.png'), '4BPP PNG (*.png)')
        self.main_gui.settings['recent.png'] = path
        if not len(path): return
        label = self.prompt_gfx_label('Import Gfx', primary)
        if label is None: return
        self.main_gui.project.import_gfx(primary, label, path)
        self.load_gfx()
        self.main_gui.update()

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
                        namespace_roots[namespace].context_data = namespace
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
            gfx_root.context, gfx_root.context_data = GFX_PRIMARY, gfx
            gfx_root.setIcon(0, QIcon(icon_paths['gfx']))
        for gfx in sorted(project.gfxs_secondary):
            gfx_root = QTreeWidgetItem(self.gfx_secondary_root, [f'{gfx}'])
            gfx_root.context, gfx_root.context_data = GFX_SECONDARY, gfx
            gfx_root.setIcon(0, QIcon(icon_paths['gfx']))


def remove_children(widget):
    """ Helper method to clear all children of a widget. """
    children = [widget.child(idx) for idx in range(widget.childCount())]
    for child in children:
        widget.removeChild(child)


