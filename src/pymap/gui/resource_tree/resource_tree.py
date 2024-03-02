"""Resource tree widget."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from pymap.gui.icon import Icon, icon_paths
from pymap.gui.properties import get_member_by_path, set_member_by_path
from pymap.gui.resource_tree.item.gfx import ResourceParameterTreeItemGfx

from .item import (
    ResourceParameterTreeItem,
    ResourceParameterTreeItemBank,
    ResourceParameterTreeItemFooter,
    ResourceParameterTreeItemFooterRoot,
    ResourceParameterTreeItemGfxRoot,
    ResourceParameterTreeItemHeader,
    ResourceParameterTreeItemHeaderRoot,
    ResourceParameterTreeItemNamespace,
    ResourceParameterTreeItemTileset,
    ResourceParameterTreeItemTilesetRoot,
)

if TYPE_CHECKING:
    from pymap.gui.gui import PymapGui


class HeaderSorting(StrEnum):
    """Enum for the sorting of the headers."""

    BANK = 'bank'
    NAMESPACE = 'namespace'


class ResourceParameterTree(QTreeWidget):
    """Widget for the resource tree."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the resource tree.

        Args:
            main_gui (PymapGui): The main gui.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent)
        self.main_gui = main_gui
        self.tree_header = ResourceParameterTreeItem(['Resource'])
        self.setHeaderItem(self.tree_header)
        self.header_root = ResourceParameterTreeItemHeaderRoot(self, ['Map Header'])
        self.header_root.setIcon(0, QIcon(icon_paths[Icon.TREE]))
        self.footer_root = ResourceParameterTreeItemFooterRoot(self, ['Map Footer'])
        self.footer_root.setIcon(0, QIcon(icon_paths[Icon.TREE]))
        self.tileset_root = ResourceParameterTreeItem(self, ['Tileset'])
        self.tileset_root.setIcon(0, QIcon(icon_paths[Icon.TREE]))
        self.tileset_primary_root = ResourceParameterTreeItemTilesetRoot(
            self.tileset_root, ['Primary'], primary=True
        )
        self.tileset_primary_root.setIcon(0, QIcon(icon_paths[Icon.FOLDER]))
        self.tileset_secondary_root = ResourceParameterTreeItemTilesetRoot(
            self.tileset_root,
            ['Secondary'],
            primary=False,
        )
        self.tileset_secondary_root.setIcon(0, QIcon(icon_paths[Icon.FOLDER]))
        self.gfx_root = ResourceParameterTreeItem(self, ['Gfx'])
        self.gfx_root.setIcon(0, QIcon(str(icon_paths[Icon.TREE])))
        self.gfx_primary_root = ResourceParameterTreeItemGfxRoot(
            self.gfx_root, ['Primary'], primary=True
        )
        self.gfx_primary_root.setIcon(0, QIcon(icon_paths[Icon.FOLDER]))
        self.gfx_secondary_root = ResourceParameterTreeItemGfxRoot(
            self.gfx_root, ['Secondary'], primary=False
        )
        self.gfx_secondary_root.setIcon(0, QIcon(icon_paths[Icon.FOLDER]))

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.context_menu_requested)

        self.itemDoubleClicked.connect(self.item_double_clicked)

    def context_menu_requested(self, position: QPoint):
        """Spawns a context menu at a given position if possible."""
        if self.main_gui.project is None:
            return
        item = self.itemAt(position)
        assert isinstance(item, ResourceParameterTreeItem)
        menu = item.context_menu(self)
        if menu is not None:
            menu.exec_(self.viewport().mapToGlobal(position))

    def item_double_clicked(self, item: QTreeWidgetItem, column_idx: int):
        """Triggered when an item is double clicked."""
        assert isinstance(item, ResourceParameterTreeItem)
        item.double_clicked(self)

    def create_bank(self, *args: Any):
        """Prompts a dialog to create a new map bank."""
        if self.main_gui.project is None:
            return
        unused_banks = self.main_gui.project.unused_banks()
        if len(unused_banks) == 0:
            return
        bank, ok_pressed = QInputDialog.getItem(
            self,
            'Create New Map Bank',
            'Enter the index of the map bank to create:',
            unused_banks,
            0,
            False,
        )
        if ok_pressed:
            self.main_gui.project.headers[bank] = {}
            self.load_headers()

    def remove_bank(self, bank: str):
        """Removes a bank from the project with prompt."""
        assert self.main_gui.project is not None
        pressed = QMessageBox.question(
            self,
            'Confirm bank removal',
            f'Do you really want to remove bank {bank} from the project entirely?',
        )
        if pressed == QMessageBox.StandardButton.Yes:
            for map_idx in self.main_gui.project.headers[bank]:
                if (
                    bank == self.main_gui.header_bank
                    and map_idx == self.main_gui.header_map_idx
                ):
                    self.main_gui.clear_header()
            self.main_gui.project.remove_bank(bank)
            self.load_headers()

    def refactor_header(
        self,
        *args: Any,
        bank: str,
        map_idx: str,
        label: str | None = None,
        namespace: str | None = None,
    ):
        """Refactors a map header."""
        if self.main_gui.project is None:
            return  #
        if label is None:
            label = self.prompt_header_label(
                f'Refactor Header [{bank}, {map_idx.zfill(2) if map_idx else "??"}]'
            )
            if label is None:
                return
        if namespace is None:
            namespace = self.prompt_namespace(
                f'Refactor Header [{bank}, {map_idx.zfill(2)}]'
            )
            if namespace is None:
                return
        self.main_gui.project.refactor_header(bank, map_idx, label, namespace)
        self.load_headers()
        self.main_gui.update()

    def prompt_header_label(self, title: str):
        """Prompts a label for a header."""
        assert self.main_gui.project is not None
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(
                self, title, 'Select a unique label for the header:'
            )
            if not ok_pressed:
                return None
            # Check if label already exists
            for other_bank in self.main_gui.project.headers:
                if label is None:
                    break
                for other_map_idx in self.main_gui.project.headers[other_bank]:
                    if label is None:
                        break
                    if (
                        label
                        == self.main_gui.project.headers[other_bank][other_map_idx][0]
                    ):
                        QMessageBox.critical(
                            self,
                            'Inavlid label',
                            (
                                f'Label {label} already exists at [{other_bank}, '
                                f'{other_map_idx.zfill(2)}]'
                            ),
                        )
                        label = None
            return label

    def prompt_namespace(self, title: str):
        """Prompts a namespace."""
        assert self.main_gui.project is not None
        (
            available_namespaces,
            namespaces_strict,
        ) = self.main_gui.project.available_namespaces()
        # Prompt the namespace
        if len(available_namespaces) > 0:
            namespace, ok_pressed = QInputDialog.getItem(
                self,
                title,
                'Select a namespace for the header:',
                available_namespaces,
                0,
                namespaces_strict,
            )
        else:
            namespace, ok_pressed = QInputDialog.getText(
                self, title, 'Enter a namespace for the header:'
            )
        if not ok_pressed:
            return None
        else:
            return namespace

    def prompt_non_full_bank(self, title: str) -> str | None:
        """Prompts a map bank that is not full yet."""
        assert self.main_gui.project is not None
        available_banks = sorted(
            [
                bank
                for bank in self.main_gui.project.headers
                if len(self.main_gui.project.unused_map_idx(bank)) > 0
            ],
            key=int,
        )
        if len(available_banks) == 0:
            QMessageBox.critical(
                self,
                'All Banks Full',
                'There is no space for new headers in any map bank.',
            )
            return None
        bank, ok_pressed = QInputDialog.getItem(
            self,
            title,
            'Select the bank to create the header in:',
            available_banks,
            0,
            False,
        )
        if not ok_pressed:
            return None
        else:
            return bank

    def prompt_unused_map_idx(self, title: str, bank: str) -> str | None:
        """Prompts an unused map idx in a given bank."""
        assert self.main_gui.project is not None
        unused_map_idx = self.main_gui.project.unused_map_idx(bank)
        if len(unused_map_idx) == 0:
            QMessageBox.critical(
                self, 'Bank Full', f'There is no space for new headers in bank {bank}.'
            )
            return None
        map_idx, ok_pressed = QInputDialog.getItem(
            self,
            title,
            f'Select the index of the header in bank {bank}:',
            unused_map_idx,
            0,
            False,
        )
        if not ok_pressed:
            return None
        else:
            return map_idx

    def prompt_footer(self, title: str):
        """Prompts a footer."""
        assert self.main_gui.project is not None
        available_footers = list(
            sorted(
                self.main_gui.project.footers,
                key=(
                    lambda footer_label: self.main_gui.project.footers[footer_label].idx  # type: ignore
                ),
            )
        )
        if len(available_footers) == 0:
            QMessageBox.critical(
                self,
                'No Footers',
                'There is no footer in the project to associate the header with.',
            )
            return None
        footer, ok_pressed = QInputDialog.getItem(
            self,
            title,
            'Select a footer this map is associated with:',
            available_footers,
            0,
            False,
        )
        if not ok_pressed:
            return None
        else:
            return footer

    def create_header(  # noqa: C901
        self, *args: Any, bank: str | None = None, namespace: str | None = None
    ):
        """Prompts a dialog to create a new header."""
        if self.main_gui.project is None:
            return
        if bank is None:
            bank = self.prompt_non_full_bank('Create New Header')
            if bank is None:
                return
        map_idx = self.prompt_unused_map_idx('Create New Header', bank)
        if map_idx is None:
            return
        footer = self.prompt_footer('Create Header')
        if footer is None:
            return
        if namespace is None:
            namespace = self.prompt_namespace('Create New Header')
            if namespace is None:
                return
        label = None
        if label is None:
            label = self.prompt_header_label('Create New Header')
            if label is None:
                return
        footer_idx = self.main_gui.project.footers[footer][0]
        # Prompt the file path and create the new file
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Create New Map Header',
            str(Path(self.main_gui.settings.settings['recent_header']).parent),
            'Pymap Structure (*.pms)',
        )
        if not len(path):
            return
        self.main_gui.settings.settings['recent_header'] = path

        # Create new map file
        header = self.main_gui.project.new_header(label, path, namespace, bank, map_idx)
        # Assign the proper namespace, footer and footer index
        set_member_by_path(
            header,
            footer,
            self.main_gui.project.config['pymap']['header']['footer_path'],
        )
        set_member_by_path(
            header,
            footer_idx,
            self.main_gui.project.config['pymap']['header']['footer_idx_path'],
        )
        set_member_by_path(
            header,
            namespace,
            self.main_gui.project.config['pymap']['header']['namespace_path'],
        )
        self.main_gui.project.save_header(header, bank, map_idx)
        self.load_headers()
        self.main_gui.update()

    def remove_header(self, *args: Any, bank: str, map_idx: str):
        """Removes a header from the project."""
        assert self.main_gui.project is not None
        pressed = QMessageBox.question(
            self,
            'Confirm header removal',
            f'Do you really want to remove header [{bank}, '
            '{map_idx.zfill(2) if map_idx else "??"}] from the project entirely?',
        )
        if pressed == QMessageBox.StandardButton.Yes:
            if (
                bank == self.main_gui.header_bank
                and map_idx == self.main_gui.header_map_idx
            ):
                self.main_gui.clear_header()
            self.main_gui.project.remove_header(bank, map_idx)
            self.load_headers()
            self.main_gui.update()

    def import_header(
        self, *args: Any, bank: str | None = None, map_idx: str | None = None
    ):
        """Imports a map header structure into the project."""
        assert self.main_gui.project is not None
        if bank is None:
            bank = self.prompt_non_full_bank('Import Header')
            if bank is None:
                return
        if map_idx is None:
            map_idx = self.prompt_unused_map_idx('Import Header', bank)
            if map_idx is None:
                return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Header',
            str(
                Path(self.main_gui.settings.settings['recent_header']).parent
                / f'{bank}_{map_idx}.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        self.main_gui.settings.settings['recent_header'] = path
        if not len(path):
            return
        label = self.prompt_header_label('Import Header')
        if label is None:
            return
        namespace = self.prompt_namespace('Import Header')
        if namespace is None:
            return
        footer = self.prompt_footer('Import Footer')
        if footer is None:
            return
        self.main_gui.project.import_header(
            bank, map_idx, label, path, namespace, footer
        )
        self.load_headers()
        self.main_gui.update()

    def prompt_unused_footer_idx(self, title: str):
        """Prompts a dialog that asks for an unused footer index."""
        assert self.main_gui.project is not None
        available_idx = list(
            map(str, sorted(list(self.main_gui.project.unused_footer_idx())))
        )
        if not available_idx:
            QMessageBox.critical(
                self,
                title,
                (
                    'There are no available footer index. Remove another footer '
                    'to add a new one first.'
                ),
            )
            return None
        footer_idx, ok_pressed = QInputDialog.getItem(
            self,
            title,
            'Select the index in the footer table:',
            available_idx,
            0,
            False,
        )
        if not ok_pressed:
            return None
        else:
            return footer_idx

    def prompt_tileset(self, title: str, primary: bool):
        """Prompts a dialog that asks for a tileset."""
        assert self.main_gui.project is not None
        tilesets = (
            list(self.main_gui.project.tilesets_primary)
            if primary
            else list(self.main_gui.project.tilesets_secondary)
        )
        if len(tilesets) == 0:
            return QMessageBox.critical(
                self,
                title,
                (
                    f'There are no {"primary" if primary else "secondary"} '
                    'tilesets to assign to the footer.'
                ),
            )
        tileset, ok_pressed = QInputDialog.getItem(
            self,
            title,
            f'Select a {"primary" if primary else "secondary"} tileset',
            tilesets,
            0,
            False,
        )
        if not ok_pressed:
            return None
        else:
            return tileset

    def prompt_footer_label(self, title: str):
        """Prompts a dialog to enter a unique label for a footer."""
        assert self.main_gui.project is not None
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(
                self, title, 'Select a unique label for the footer:'
            )
            if not ok_pressed:
                return None
            if label in self.main_gui.project.footers:
                QMessageBox.critical(
                    self,
                    'Invalid label',
                    f'The label {label} is already used for another footer.',
                )
                label = None
        return label

    def create_footer(self, *args: Any):
        """Prompts a dialog to create a new map footer."""
        if self.main_gui.project is None:
            return
        footer_idx = self.prompt_unused_footer_idx('Create New Footer')
        if footer_idx is None:
            return
        tileset_primary = self.prompt_tileset('Create New Footer', True)
        if tileset_primary is None:
            return
        tileset_secondary = self.prompt_tileset('Create New Footer', False)
        if tileset_secondary is None:
            return
        label = self.prompt_footer_label('Create New Footer')
        if label is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Create New Footer',
            str(
                Path(self.main_gui.settings.settings['recent_footer']).parent
                / f'{label}.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        if not len(path):
            return
        self.main_gui.settings.settings['recent_footer'] = path

        # Create new footer
        footer = self.main_gui.project.new_footer(label, path, int(footer_idx))
        # Assign the tilesets
        set_member_by_path(
            footer,
            tileset_primary,
            self.main_gui.project.config['pymap']['footer']['tileset_primary_path'],
        )
        set_member_by_path(
            footer,
            tileset_secondary,
            self.main_gui.project.config['pymap']['footer']['tileset_secondary_path'],
        )
        self.main_gui.project.save_footer(footer, label)
        self.load_footers()
        self.main_gui.update()

    def remove_footer(self, *args: Any, footer: str):
        """Removes a footer from the project with prompt."""
        assert self.main_gui.project is not None
        pressed = QMessageBox.question(
            self,
            'Confirm footer removal',
            f'Do you really want to remove footer {footer} from the project entirely?',
        )
        if pressed == QMessageBox.StandardButton.Yes:
            # Scan through the entire project and change all references to this
            # footer to None
            headers: list[tuple[str, str]] = []
            for bank in self.main_gui.project.headers:
                for map_idx in self.main_gui.project.headers[bank]:
                    if (
                        map_idx == self.main_gui.header_map_idx
                        and bank == self.main_gui.header_bank
                    ):
                        if self.main_gui.footer_label == footer:
                            headers.append((bank, map_idx))
                    else:
                        header, _, _ = self.main_gui.project.load_header(bank, map_idx)
                        if footer == get_member_by_path(
                            header,
                            self.main_gui.project.config['pymap']['header'][
                                'footer_path'
                            ],
                        ):
                            headers.append((bank, map_idx))
            if len(headers) > 0:
                headers_readable = [
                    (
                        f'[{header[0]}, {header[1].zfill(2)}] '
                        f'{self.main_gui.project.headers[header[0]][header[1]][0]}'
                    )
                    for header in headers
                ]
                return QMessageBox.critical(
                    self,
                    'Confirm footer removal',
                    (
                        f'The following headers refer to footer {footer}: '
                        f'{", ".join(headers_readable)}. Assign different footers '
                        'to those headers first.'
                    ),
                )
            self.main_gui.project.remove_footer(footer)
            self.load_footers()
            self.main_gui.update()

    def refactor_footer(self, *args: Any, label_old: str, label_new: str | None = None):
        """Refactors the map footer's label."""
        if self.main_gui.project is None:
            return
        if label_new is None:
            label_new = self.prompt_footer_label('Relabel Map Footer')
            if label_new is None:
                return
        self.main_gui.project.refactor_footer(label_old, label_new)
        # If the current map refers to this footer, change the current label as well
        if self.main_gui.footer_label == label_old:
            self.main_gui.footer_label = label_new
            set_member_by_path(
                self.main_gui.header,
                label_new,
                self.main_gui.project.config['pymap']['header']['footer_path'],
            )
        self.load_footers()
        self.main_gui.update()

    def import_footer(self, *args: Any):
        """Imports a map header structure into the project."""
        assert self.main_gui.project is not None
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Footer',
            str(
                Path(self.main_gui.settings.settings['recent_footer']).parent
                / 'footer.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        self.main_gui.settings.settings['recent_footer'] = path
        if not len(path):
            return
        footer_idx = self.prompt_unused_footer_idx('Import Footer')
        if footer_idx is None:
            return
        label = self.prompt_footer_label('Import Footer')
        if label is None:
            return
        self.main_gui.project.import_footer(label, path, int(footer_idx))
        self.load_footers()
        self.main_gui.update()

    def prompt_gfx(self, title: str, primary: bool):
        """Prompts for a gfx by a dialog."""
        assert self.main_gui.project is not None
        gfxs = (
            self.main_gui.project.gfxs_primary
            if primary
            else self.main_gui.project.gfxs_secondary
        )
        if len(gfxs) == 0:
            QMessageBox.critical(
                self, title, 'There are no gfxs avaialable for this tileset type.'
            )
            return None
        gfx, ok_pressed = QInputDialog.getItem(
            self, title, 'Select a gfx the tileset uses:', list(gfxs.keys()), 0, False
        )
        if not ok_pressed:
            return None
        return gfx

    def prompt_tileset_label(self, title: str, primary: bool):
        """Prompts for a label of a tileset."""
        assert self.main_gui.project is not None
        tilesets = (
            self.main_gui.project.tilesets_primary
            if primary
            else self.main_gui.project.tilesets_secondary
        )
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(
                self, title, 'Select a unique label for the tileset:'
            )
            if not ok_pressed:
                return None
            if label in tilesets:
                QMessageBox.critical(
                    self,
                    title,
                    (
                        f'The label {label} is already used for another tileset '
                        'of this type.'
                    ),
                )
                label = None
        return label

    def prompt_gfx_label(self, title: str, primary: bool, text: str):
        """Prompts for an unused gfx label."""
        assert self.main_gui.project is not None
        gfxs = (
            self.main_gui.project.gfxs_primary
            if primary
            else self.main_gui.project.gfxs_secondary
        )
        label = None
        while label is None:
            label, ok_pressed = QInputDialog.getText(
                self, title, 'Select a unique label for the gfx:', text=text
            )
            if not ok_pressed:
                return None
            if label in gfxs:
                QMessageBox.critical(
                    self,
                    title,
                    f'The label {label} is already used for another gfx of this type.',
                )
                label = None
        return label

    def create_tileset(self, *args: Any, primary: bool):
        """Prompts a dialog to create a new tileset."""
        if self.main_gui.project is None:
            return
        gfx = self.prompt_gfx('Create Tileset', primary)
        if gfx is None:
            return
        label = self.prompt_tileset_label('Create Tileset', primary)
        if label is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Create Tileset',
            str(
                Path(self.main_gui.settings.settings['recent_tileset']).parent
                / f'{label}.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        if not len(path):
            return
        self.main_gui.settings.settings['recent_tileset'] = path

        # Create new tileset and assign the gfx
        tileset = self.main_gui.project.new_tileset(primary, label, path)
        set_member_by_path(
            tileset,
            gfx,
            self.main_gui.project.config['pymap'][
                'tileset_primary' if primary else 'tileset_secondary'
            ]['gfx_path'],
        )
        self.main_gui.project.save_tileset(primary, tileset, label)
        self.load_tilesets()
        self.main_gui.update()

    def duplicate_tileset(
        self, src_label: str, primary: bool, label: str | None = None
    ):
        """Duplicates a tileset."""
        if self.main_gui.project is None:
            return
        label = self.prompt_tileset_label(
            f'Duplicate tileset Tileset {src_label}', primary
        )
        if label is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save duplicate of Tileset {src_label}',
            str(
                Path(self.main_gui.settings.settings['recent_tileset']).parent
                / f'{label}.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        if not len(path):
            return
        self.main_gui.settings.settings['recent_tileset'] = path

        tileset = deepcopy(self.main_gui.project.load_tileset(primary, src_label))
        self.main_gui.project.new_tileset(primary, label, path, tileset=tileset)
        self.load_tilesets()
        self.main_gui.update()

    def remove_tileset(self, *args: Any, primary: bool, label: str):
        """Removes a tileset."""
        assert self.main_gui.project is not None
        pressed = QMessageBox.question(
            self,
            'Confirm tileset removal',
            f'Do you really want to remove tileset {label} from the project entirely?',
        )
        if pressed == QMessageBox.StandardButton.Yes:
            # Scan through all footers and collect footers that refer to this
            # primary / secondary tileset
            footers: list[str] = []
            for footer_label in self.main_gui.project.footers:
                if footer_label == self.main_gui.footer_label:
                    if (label == self.main_gui.tileset_primary_label and primary) or (
                        label == self.main_gui.tileset_secondary_label and not primary
                    ):
                        footers.append(footer_label)
                else:
                    footer, _ = self.main_gui.project.load_footer(footer_label)
                    if label == get_member_by_path(
                        footer,
                        self.main_gui.project.config['pymap']['footer'][
                            'tileset_primary_path'
                            if primary
                            else 'tileset_secondary_path'
                        ],
                    ):
                        footers.append(footer_label)
            if len(footers) > 0:
                return QMessageBox.critical(
                    self,
                    'Tileset Removal',
                    (
                        f'The following footers refer to the tileset {label}: '
                        f'{", ".join(footers)}. '
                        'Assign different tilesets to those footers first.'
                    ),
                )
            self.main_gui.project.remove_tileset(primary, label)
            self.load_tilesets()
            self.main_gui.update()

    def refactor_tileset(
        self,
        *args: Any,
        primary: bool,
        label_old: str,
        label_new: str | None = None,
    ):
        """Changes the label of a tileset."""
        if self.main_gui.project is None:
            return
        if label_new is None:
            label_new = self.prompt_tileset_label('Relabel Tileset', primary)
            if label_new is None:
                return
        self.main_gui.project.refactor_tileset(primary, label_old, label_new)
        # If the current footer refers to this tileset, change the label as well
        if primary and self.main_gui.tileset_primary_label == label_old:
            self.main_gui.tileset_primary_label = label_new
            set_member_by_path(
                self.main_gui.footer,
                label_new,
                self.main_gui.project.config['pymap']['footer']['tileset_primary_path'],
            )
        elif not primary and self.main_gui.tileset_secondary_label == label_old:
            self.main_gui.tileset_secondary_label = label_new
            set_member_by_path(
                self.main_gui.footer,
                label_new,
                self.main_gui.project.config['pymap']['footer'][
                    'tileset_secondary_path'
                ],
            )
        self.load_tilesets()
        self.main_gui.update()

    def import_tileset(self, *args: Any, primary: bool):
        """Imports a tileset."""
        if self.main_gui.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Tileset',
            str(
                Path(self.main_gui.settings.settings['recent_tileset']).parent
                / 'tileset.pms'
            ),
            'Pymap Structure (*.pms)',
        )
        self.main_gui.settings.settings['recent_tileset'] = path
        if not len(path):
            return
        label = self.prompt_tileset_label('Import Tileset', primary)
        if label is None:
            return
        self.main_gui.project.import_tileset(primary, label, path)
        self.load_tilesets()
        self.main_gui.update()

    def refactor_gfx(
        self,
        *args: Any,
        primary: bool,
        label_old: str,
        label_new: str | None = None,
    ):
        """Changes the label of a gfx."""
        if self.main_gui.project is None:
            return
        if label_new is None:
            label_new = self.prompt_gfx_label(
                'Relabel Gfx', primary, 'Select a new label for the gfx:'
            )
            if label_new is None:
                return
        self.main_gui.project.refactor_gfx(primary, label_old, label_new)
        # If the current tileset refers to this gfx, change the label as well
        if primary:
            gfx_primary_label = get_member_by_path(
                self.main_gui.tileset_primary,
                self.main_gui.project.config['pymap']['tileset_primary']['gfx_path'],
            )
            if gfx_primary_label == label_old:
                set_member_by_path(
                    self.main_gui.tileset_primary,
                    label_new,
                    self.main_gui.project.config['pymap']['tileset_primary'][
                        'gfx_path'
                    ],
                )
        else:
            gfx_secondary_label = get_member_by_path(
                self.main_gui.tileset_secondary,
                self.main_gui.project.config['pymap']['tileset_secondary']['gfx_path'],
            )
            if gfx_secondary_label == label_old:
                set_member_by_path(
                    self.main_gui.tileset_secondary,
                    label_new,
                    self.main_gui.project.config['pymap']['tileset_secondary'][
                        'gfx_path'
                    ],
                )
        self.load_gfx()
        self.main_gui.update()

    def remove_gfx(self, *args: Any, primary: bool, label: str):
        """Removes a gfxs."""
        assert self.main_gui.project is not None
        pressed = QMessageBox.question(
            self,
            'Confirm gfx removal',
            f'Do you really want to remove gfx {label} from the project entirely?',
        )
        if pressed == QMessageBox.StandardButton.Yes:
            # Scan through all tilesets and collect tilesets that refer to this gfx
            tilesets: list[str] = []
            for tileset_label in (
                self.main_gui.project.tilesets_primary
                if primary
                else self.main_gui.project.tilesets_secondary
            ):
                # Check if currently the active tileset in display is refering to
                # this gfx
                if (
                    self.main_gui.tileset_primary_label == tileset_label and primary
                ) or (
                    self.main_gui.tileset_secondary_label == tileset_label
                    and not primary
                ):
                    tileset = (
                        self.main_gui.tileset_primary
                        if primary
                        else self.main_gui.tileset_secondary
                    )
                else:
                    tileset = self.main_gui.project.load_tileset(primary, tileset_label)
                if label == get_member_by_path(
                    tileset,
                    self.main_gui.project.config['pymap'][
                        'tileset_primary' if primary else 'tileset_secondary'
                    ]['gfx_path'],
                ):
                    tilesets.append(tileset_label)
            if len(tilesets) > 0:
                return QMessageBox.critical(
                    self,
                    'Gfx Removal',
                    (
                        f'The following tilesets refer to the gfx {label}: '
                        f'{", ".join(tilesets)}. Assign different gfxs to '
                        'those tilesets first.'
                    ),
                )
            self.main_gui.project.remove_gfx(primary, label)
            self.load_gfx()
            self.main_gui.update()

    def import_gfx(self, *args: Any, primary: bool):
        """Imports a gfx."""
        if self.main_gui.project is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Import Gfx',
            str(
                Path(self.main_gui.settings.settings['recent_gfx']).parent
                / 'tileset.png'
            ),
            '4BPP PNG (*.png)',
        )
        self.main_gui.settings.settings['recent_gfx'] = path
        if not len(path):
            return
        label = self.prompt_gfx_label(
            'Import Gfx', primary, 'Select a label for the gfx:'
        )
        if label is None:
            return
        self.main_gui.project.import_gfx(primary, label, path)
        self.load_gfx()
        self.main_gui.update()

    def load_project(self):
        """Updates the tree of a project."""
        self.load_headers()
        self.load_footers()
        self.load_tilesets()
        self.load_gfx()

    def load_headers(self):
        """Updates the headers of a project."""
        assert self.main_gui.project is not None
        project = self.main_gui.project
        sort_headers = self.main_gui.settings.settings['resource_tree_header_listing']
        # Remove old headers
        remove_children(self.header_root)
        # Add new headers
        if sort_headers == HeaderSorting.BANK:
            for bank in sorted(project.headers, key=int):
                bank_root = ResourceParameterTreeItemBank(
                    self.header_root, [f'Bank {bank}'], bank=bank
                )
                bank_root.setIcon(0, QIcon(icon_paths[Icon.FOLDER]))
                for map_idx in sorted(project.headers[bank], key=int):
                    label, _, namespace = project.headers[bank][map_idx]
                    map_root = ResourceParameterTreeItemHeader(
                        bank_root,
                        [f'[{bank}, {map_idx.zfill(2)}] {label}'],
                        bank=bank,
                        map_idx=map_idx,
                    )
                    map_root.setIcon(0, QIcon(icon_paths[Icon.HEADER]))
        elif sort_headers == HeaderSorting.NAMESPACE:
            namespace_roots: dict[str, ResourceParameterTreeItemNamespace] = {}
            for bank in sorted(project.headers, key=int):
                for map_idx in sorted(project.headers[bank], key=int):
                    label, _, namespace = project.headers[bank][map_idx]
                    if namespace not in namespace_roots:
                        assert namespace is not None
                        namespace_roots[namespace] = ResourceParameterTreeItemNamespace(
                            self.header_root, [str(namespace)], namespace=namespace
                        )
                        namespace_roots[namespace].setIcon(
                            0, QIcon(icon_paths[Icon.FOLDER])
                        )
                    map_root = ResourceParameterTreeItemHeader(
                        namespace_roots[namespace],
                        [f'[{bank}, {map_idx.zfill(2)}] {label}'],
                        bank=bank,
                        map_idx=map_idx,
                    )
                    map_root.setIcon(0, QIcon(icon_paths[Icon.HEADER]))

    def load_footers(self):
        """Updates the footers of a project."""
        project = self.main_gui.project
        assert project is not None
        # Remove old footers
        remove_children(self.footer_root)
        # Add new footers
        for footer in sorted(
            project.footers, key=(lambda label: project.footers[label][0])
        ):
            footer_idx, _ = project.footers[footer]
            footer_root = ResourceParameterTreeItemFooter(
                self.footer_root,
                [f'[{str(footer_idx).zfill(3)}] {footer}'],
                label=footer,
            )
            footer_root.setIcon(0, QIcon(icon_paths[Icon.FOOTER]))

    def load_tilesets(self):
        """Updates the tilesets of a project."""
        project = self.main_gui.project
        assert project is not None
        # Remove old tilesets
        remove_children(self.tileset_primary_root)
        remove_children(self.tileset_secondary_root)
        # Add new tilesets
        for tileset in sorted(project.tilesets_primary):
            tileset_root = ResourceParameterTreeItemTileset(
                self.tileset_primary_root, [f'{tileset}'], primary=True, label=tileset
            )
            tileset_root.setIcon(0, QIcon(icon_paths[Icon.TILESET]))
        for tileset in sorted(project.tilesets_secondary):
            tileset_root = ResourceParameterTreeItemTileset(
                self.tileset_secondary_root,
                [f'{tileset}'],
                primary=False,
                label=tileset,
            )
            tileset_root.setIcon(0, QIcon(icon_paths[Icon.TILESET]))

    def load_gfx(self):
        """Updates the gfxs of a project."""
        project = self.main_gui.project
        assert project is not None
        # Remove old gfx
        remove_children(self.gfx_primary_root)
        remove_children(self.gfx_secondary_root)
        for gfx in sorted(project.gfxs_primary):
            gfx_root = ResourceParameterTreeItemGfx(
                self.gfx_primary_root, [f'{gfx}'], primary=True, label=gfx
            )
            gfx_root.setIcon(0, QIcon(icon_paths[Icon.GFX]))
        for gfx in sorted(project.gfxs_secondary):
            gfx_root = ResourceParameterTreeItemGfx(
                self.gfx_secondary_root, [f'{gfx}'], primary=False, label=gfx
            )
            gfx_root.setIcon(0, QIcon(icon_paths[Icon.GFX]))


def remove_children(widget: QTreeWidgetItem):
    """Helper method to clear all children of a widget."""
    children = [widget.child(idx) for idx in range(widget.childCount())]
    for child in children:
        widget.removeChild(child)
