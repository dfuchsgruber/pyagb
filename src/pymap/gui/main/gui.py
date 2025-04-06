"""Widget for the main gui."""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPainter, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTabWidget,
    QWidget,
)
from skimage.measure import label as label_image  # type: ignore

from pymap.gui import render
from pymap.gui.connection import ConnectionWidget
from pymap.gui.event import EventWidget
from pymap.gui.footer import FooterWidget
from pymap.gui.header import HeaderWidget
from pymap.gui.history import (
    AssignFooter,
    AssignGfx,
    AssignTileset,
    ReplaceBlocks,
    ResizeBorder,
    ResizeMap,
    SetBlocks,
    SetBorder,
)
from pymap.gui.history.smart_shape import SetSmartShapeBlocks, SmartShapeReplaceBlocks
from pymap.gui.main.open_history import OpenHistory, OpenHistoryItem
from pymap.gui.map import MapWidget
from pymap.gui.resource_tree import HeaderSorting, ResourceParameterTree
from pymap.gui.smart_shape.smart_shape import SmartShape
from pymap.gui.tileset import TilesetWidget
from pymap.gui.types import MapLayers, RGBAImage, Tilemap
from pymap.project import Project

from .model import PymapGuiModel


class PymapGui(QMainWindow, PymapGuiModel):
    """Main GUI for Pymap."""

    def __init__(self, parent: QWidget | None = None):
        """Initializes the main GUI.

        Args:
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent)
        PymapGuiModel.__init__(self)
        self.settings = QSettings('dfuchsgruber', 'pymap')
        self.open_history = OpenHistory(self)

        # Add the project tree widget
        self.resource_tree_widget = QDockWidget('Project Resources')
        self.resource_tree = ResourceParameterTree(self)
        self.resource_tree_widget.setWidget(self.resource_tree)
        self.resource_tree_widget.setFloating(False)
        self.resource_tree_widget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.resource_tree_widget
        )

        # Add the tabs
        self.central_widget = QTabWidget()
        self.event_widget = EventWidget(self)
        self.map_widget = MapWidget(self)
        self.connection_widget = ConnectionWidget(self)
        self.header_widget = HeaderWidget(self)
        self.footer_widget = FooterWidget(self)
        self.tileset_widget = TilesetWidget(self)

        self.central_widget.addTab(self.map_widget, 'Map')
        self.central_widget.addTab(self.event_widget, 'Events')
        self.central_widget.addTab(self.tileset_widget, 'Tileset')
        self.central_widget.addTab(self.connection_widget, 'Connections')
        self.central_widget.addTab(self.header_widget, 'Header')
        self.central_widget.addTab(self.footer_widget, 'Footer')
        self.central_widget.currentChanged.connect(self.tab_changed)

        # Build the menu bar
        # 'File' menu
        file_menu = self.menuBar().addMenu('&File')
        # 'New' submenu
        file_menu_new_menu = file_menu.addMenu('&New')
        file_menu_new_project_action = file_menu_new_menu.addAction('Project')  # type: ignore
        file_menu_new_project_action.setShortcut('Ctrl+N')
        file_menu_new_bank_action = file_menu_new_menu.addAction('Bank')  # type: ignore
        file_menu_new_bank_action.triggered.connect(self.resource_tree.create_bank)
        file_menu_new_header_action = file_menu_new_menu.addAction('Header')  # type: ignore
        file_menu_new_header_action.triggered.connect(self.resource_tree.create_header)
        file_menu_new_footer_action = file_menu_new_menu.addAction('Footer')  # type: ignore
        file_menu_new_footer_action.triggered.connect(self.resource_tree.create_footer)
        file_menu_new_tileset_action = file_menu_new_menu.addAction('Tileset')  # type: ignore
        file_menu_new_tileset_action.triggered.connect(
            self.resource_tree.create_tileset
        )
        # Flat actions
        file_menu_open_action = file_menu.addAction('&Open Project')  # type: ignore
        file_menu_open_action.triggered.connect(self.prompt_open_project)
        file_menu_open_action.setShortcut('Ctrl+O')
        self.file_menu_open_recent_menu = file_menu.addMenu('Open Recent')  # type: ignore
        self.build_open_recent_menu(self.file_menu_open_recent_menu)

        # 'Save' submenu
        file_menu_save_menu = file_menu.addMenu('&Save')
        file_menu_save_all = file_menu_save_menu.addAction('All')  # type: ignore
        file_menu_save_all.triggered.connect(self.save_all)
        file_menu_save_all.setShortcut('Ctrl+S')
        file_menu_save_project = file_menu_save_menu.addAction('Project')  # type: ignore
        file_menu_save_project.triggered.connect(self.save_project)
        file_menu_save_header = file_menu_save_menu.addAction('Header')  # type: ignore
        file_menu_save_header.triggered.connect(self.save_header)
        file_menu_save_footer = file_menu_save_menu.addAction('Footer')  # type: ignore
        file_menu_save_footer.triggered.connect(self.save_footer)
        file_menu_save_tilesets = file_menu_save_menu.addAction('Tilesets')  # type: ignore
        file_menu_save_tilesets.triggered.connect(self.save_tilesets)
        # 'Edit' menu
        edit_menu = self.menuBar().addMenu('&Edit')

        edit_menu_undo_action = edit_menu.addAction('Undo')  # type: ignore
        edit_menu_undo_action.triggered.connect(self.undo)
        edit_menu_undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        edit_menu_undo_action.setEnabled(False)
        self.edit_menu_undo_action = edit_menu_undo_action

        edit_menu_redo_action = edit_menu.addAction('Redo')  # type: ignore
        edit_menu_redo_action.triggered.connect(self.redo)
        edit_menu_redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu_redo_action.setEnabled(False)
        self.edit_menu_redo_action = edit_menu_redo_action
        edit_menu.addSeparator()

        edit_menu_resize_map_action = edit_menu.addAction('Resize Map')  # type: ignore
        edit_menu_resize_map_action.triggered.connect(self.prompt_resize_map)
        edit_menu_resize_border_action = edit_menu.addAction('Resize Border')  # type: ignore
        edit_menu_resize_border_action.triggered.connect(self.prompt_resize_border)
        edit_menu_change_tileset_submenu = edit_menu.addMenu('Change Tileset')
        edit_menu_change_tileset_primary_action = (
            edit_menu_change_tileset_submenu.addAction('Primary')  # type: ignore
        )
        edit_menu_change_tileset_primary_action.triggered.connect(
            partial(self.prompt_change_tileset, primary=True)
        )
        edit_menu_change_tileset_secondary_action = (
            edit_menu_change_tileset_submenu.addAction('Secondary')  # type: ignore
        )
        edit_menu_change_tileset_secondary_action.triggered.connect(
            partial(self.prompt_change_tileset, primary=False)
        )

        edit_menu_shift_submenu = edit_menu.addMenu('Shift')
        edit_menu_shift_blocks_and_events_action = edit_menu_shift_submenu.addAction(  # type: ignore
            'Shift Blocks and Events'
        )
        edit_menu_shift_blocks_and_events_action.triggered.connect(
            lambda: self.prompt_shift_blocks_and_events(
                shift_blocks=True, shift_events=True
            )
        )
        edit_menu_shift_blocks_action = edit_menu_shift_submenu.addAction(  # type: ignore
            'Shift Blocks'
        )
        edit_menu_shift_blocks_action.triggered.connect(
            lambda: self.prompt_shift_blocks_and_events(
                shift_blocks=True, shift_events=False
            )
        )
        edit_menu_shift_events_action = edit_menu_shift_submenu.addAction(  # type: ignore
            'Shift Events'
        )
        edit_menu_shift_events_action.triggered.connect(
            lambda: self.prompt_shift_blocks_and_events(
                shift_blocks=False, shift_events=True
            )
        )
        edit_menu.setToolTipsVisible(True)

        # 'View' menu
        view_menu = self.menuBar().addMenu('&View')
        view_menu_resource_action = view_menu.addAction('Toggle Header Listing')  # type: ignore
        view_menu_resource_action.setShortcut('Ctrl+L')
        view_menu_resource_action.triggered.connect(
            self.resource_tree_toggle_header_listing
        )
        view_menu_event_action = view_menu.addAction('Toggle Event Pictures')  # type: ignore
        view_menu_event_action.triggered.connect(self.event_widget_toggle_pictures)
        view_menu_grid_action = view_menu.addAction('Toggle Grid')  # type: ignore
        view_menu_grid_action.setShortcut('Ctrl+G')
        view_menu_grid_action.triggered.connect(self.toggle_grid)

        # 'Tools' menu
        tools_menu = self.menuBar().addMenu('Tools')
        view_menu_save_image_action = tools_menu.addAction('Save Map Image')  # type: ignore
        view_menu_save_image_action.triggered.connect(self.save_map_image)

        self.setCentralWidget(self.central_widget)

    def build_open_recent_menu(self, menu: QMenu):
        """Builds the open recent menu.

        Args:
            menu (QMenu): The menu to build.
        """
        menu.clear()
        for idx, item in enumerate(self.open_history):
            action = menu.addAction(  # type: ignore
                f'{Path(item["project_path"]).stem} - {item["label"]}'
            )
            action.triggered.connect(
                partial(
                    self.open_project_and_header,
                    path_str=item['project_path'],
                    bank=item['bank'],
                    map_idx=item['map_idx'],
                )
            )  # type: ignore
            action.setShortcut(QKeySequence(f'Ctrl+{idx}'))
        menu.addSeparator()
        action = menu.addAction('Clear History')  # type: ignore
        action.triggered.connect(self.open_history.clear)

    def undo(self):
        """Undo the last action."""
        self.central_widget.currentWidget().undo_stack.undo()  # type: ignore

    def redo(self):
        """Redo the last action."""
        self.central_widget.currentWidget().undo_stack.redo()  # type: ignore

    def update_redo_undo_tooltips(self, widget: QWidget, undo_stack: QUndoStack):
        """Updates the redo and undo tooltips."""
        if self.central_widget.currentWidget() is widget:
            if undo_stack.canUndo():
                self.edit_menu_undo_action.setEnabled(True)
                self.edit_menu_undo_action.setToolTip(undo_stack.undoText())
            else:
                self.edit_menu_undo_action.setEnabled(False)
                self.edit_menu_undo_action.setToolTip('')
            if undo_stack.canRedo():
                self.edit_menu_redo_action.setEnabled(True)
                self.edit_menu_redo_action.setToolTip(undo_stack.redoText())
            else:
                self.edit_menu_redo_action.setEnabled(False)
                self.edit_menu_redo_action.setToolTip('')

    @property
    def show_borders(self) -> bool:
        """Returns whether the borders are shown."""
        return self.map_widget.blocks_tab.show_border.isChecked()

    def tab_changed(self, *args: Any, **kwargs: Any):
        """Callback method for when a tab is changed."""
        # Update map data and blocks lazily in order to prevent lag
        # when mapping blocks or tiles
        if self.central_widget.currentWidget() is self.event_widget:
            self.map_widget.load_header()
            self.event_widget.load_header()
        if self.central_widget.currentWidget() is self.map_widget:
            self.map_widget.load_header()
        if self.central_widget.currentWidget() is self.connection_widget:
            self.connection_widget.load_header()

    def closeEvent(self, event: QCloseEvent):
        """Query to save currently open files on closing."""
        if (
            self.prompt_save_header()
            or self.prompt_save_footer()
            or self.prompt_save_tilesets()
        ):
            event.ignore()
            return
        super().closeEvent(event)

    def save_all(self):
        """Saves project, header, footer and tilesets."""
        self.save_project()
        self.save_header()
        self.save_footer()
        self.save_tilesets()

    def save_project(self):
        """Saves the current project."""
        if self.project is None or self.project_path is None:
            return
        self.project.save(self.project_path)

    def prompt_open_project(self):
        """Prompts a dialog to open a new project file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Open project',
            self.settings.value('project/recent', '.', str),  # type: ignore
            'Pymap projects (*.pmp)',
        )
        if len(path):
            self.open_project(path)

    def open_project_and_header(self, path_str: str, bank: str, map_idx: str):
        """Opens a new project file and a header.

        Args:
            path_str (str): The path to the project file.
            bank (str): The bank of the header.
            map_idx (str): The index of the header.
        """
        if self.project_path != Path(path_str):
            self.open_project(path_str)
        self.open_header(bank, map_idx, prompt_saving=True)

    def open_project(self, path_str: str):
        """Opens a new project file.

        Args:
            path_str (str): The path to the project file.
        """
        path = Path(path_str)
        os.chdir(path.parent)
        self.project_path: Path | None = path
        self.settings.setValue('project/recent', str(path.absolute()))
        self.project = Project(path)
        self.resource_tree.load_project()
        self.map_widget.load_project()
        self.footer_widget.load()
        self.header_widget.load()
        self.event_widget.load_project()
        self.tileset_widget.load_project()

    def clear_header(self):
        """Unassigns the current header, footer, tilesets."""
        self.header = None
        self.header_bank = None
        self.header_map_idx = None
        # Render subwidgets
        self.update_gui()

    def prompt_resize_map(self):
        """Prompts the user to enter new map dimensions."""
        if self.project is None or self.header is None or self.footer is None:
            return False
        width, height = self.get_map_dimensions()
        text, ok_pressed = QInputDialog.getText(
            self,
            'Resize Map',
            'Enter new map dimensions in the format "width, height"',
            text=f'{width}, {height}',
        )
        if ok_pressed:
            try:
                width_new, height_new = (
                    int(value.strip()) for value in text.split(',')
                )
            except Exception:
                QMessageBox.critical(
                    self,
                    'Invalid format',
                    'Enter new map dimensions as comma separated values!',
                )
                return
            if height_new != height or width_new != width:
                self.resize_map(height_new, width_new)

    def prompt_resize_border(self):
        """Prompts the user to enter new border dimensions."""
        if self.project is None or self.header is None or self.footer is None:
            return False
        width, height = self.get_border_dimensions()
        text, ok_pressed = QInputDialog.getText(
            self,
            'Resize Border',
            'Enter new border dimensions in the format "width, height"',
            text=f'{width}, {height}',
        )
        if ok_pressed:
            try:
                width_new, height_new = (
                    int(value.strip()) for value in text.split(',')
                )
            except Exception:
                QMessageBox.critical(
                    self,
                    'Invalid format',
                    'Enter new border dimensions as comma separated values!',
                )
                return
            if height_new != height or width_new != width:
                self.resize_border(height_new, width_new)

    def prompt_change_tileset(self, primary: bool):
        """Prompts the user to enter new tileset labels."""
        if self.project is None or self.header is None or self.footer is None:
            return False
        # Use a QInputDialog to select from self.project.tilesets_primary
        # as a combo box, only allowing choices from the list
        title = 'Primary' if primary else 'Secondary'
        choices: list[str] = list(
            self.project.tilesets_primary
            if primary
            else self.project.tilesets_secondary
        )
        text, ok_pressed = QInputDialog.getItem(
            self,
            f'Change {title} Tileset',
            f'Select a new {title} tileset',
            choices,
            editable=False,
        )
        if ok_pressed:
            self.change_tileset(text, primary)

    def prompt_shift_blocks_and_events(
        self, shift_blocks: bool = False, shift_events: bool = False
    ):
        """Prompts the user to enter by how much all blocks and events are shifted."""
        if self.project is None or self.header is None or self.footer is None:
            return False
        if shift_blocks and shift_events:
            title = 'Shift blocks and events'
        elif shift_blocks:
            title = 'Shift blocks'
        elif shift_events:
            title = 'Shift events'
        else:
            raise RuntimeError('Either shift blocks or events or both!')
        text, ok_pressed = QInputDialog.getText(
            self, title, 'Enter by how much to shift in the format "x, y"'
        )
        if ok_pressed:
            try:
                x, y = (int(value.strip()) for value in text.split(','))
            except Exception:
                QMessageBox.critical(
                    self,
                    'Invalid format',
                    'Enter by how much to shift as comma separated values!',
                )
                return
            if shift_blocks:
                self.shift_blocks(x, y)
            if shift_events:
                self.event_widget.shift_events(x, y)

    def prompt_save_header(self) -> bool:
        """Prompts to save the header if it is unsafed.

        Returns:
            bool: Whether the header saving was explicitly canceled.
        """
        if not self.header_loaded:
            return False
        if self.header is not None and (
            not self.header_widget.undo_stack.isClean()
            or not self.event_widget.undo_stack.isClean()
            or not self.connection_widget.undo_stack.isClean()
        ):
            assert self.project is not None
            assert self.header_bank is not None
            assert self.header_map_idx is not None
            pressed = self.prompt_saving(
                self,
                'Save Header Changes',
                (
                    'Header '
                    f'{self.project.headers[self.header_bank][self.header_map_idx][0]} '
                    'has changed. Do you want to save changes?'
                ),
            )
            if pressed == QMessageBox.StandardButton.Save:
                self.save_header()
            return pressed == QMessageBox.StandardButton.Cancel
        return False

    def prompt_save_footer(self) -> bool:
        """Prompts to save the footer if it is unsafed.

        Returns:
            bool: Whether the footer saving was explicitly canceled.
        """
        if (
            self.footer_loaded
            and self.footer is not None
            and (
                not self.map_widget.undo_stack.isClean()
                or not self.footer_widget.undo_stack.isClean()
            )
        ):
            pressed = self.prompt_saving(
                self,
                'Save Footer Changes',
                f'Footer {self.footer_label} has changed. Do you want to save changes?',
            )
            if pressed == QMessageBox.StandardButton.Save:
                self.save_footer()
            return pressed == QMessageBox.StandardButton.Cancel
        return False

    def prompt_save_tilesets(self) -> bool:
        """Prompts to save the tilesets if they are unsafed.

        Returns:
            bool: Whether the tileset saving was explicitly canceled.
        """
        if self.tilesets_loaded and not self.tileset_widget.undo_stack.isClean():
            pressed = self.prompt_saving(
                self,
                'Save Tileset Changes',
                (
                    f'Tilesets {self.tileset_primary_label} '
                    f'and {self.tileset_secondary_label} have changed. '
                    'Do you want to save changes?'
                ),
            )
            if pressed == QMessageBox.StandardButton.Save:
                self.save_tilesets()
            return pressed == QMessageBox.StandardButton.Cancel
        return False

    # @Profile('open_header')
    def open_header(self, bank: str, map_idx: str, prompt_saving: bool = True):
        """Opens a new map header and displays it."""
        if self.project is None:
            return
        if prompt_saving and (
            self.prompt_save_header()
            or self.prompt_save_footer()
            or self.prompt_save_tilesets()
        ):
            return
        self.header_widget.undo_stack.clear()
        self.event_widget.undo_stack.clear()
        self.connection_widget.undo_stack.clear()
        self.header, _, _ = self.project.load_header(bank, map_idx)
        self.header_bank = bank
        self.header_map_idx = map_idx
        # Trigger opening of the footer
        self.open_footer(self.get_footer_label(), prompt_saving=False)
        self.resource_tree.select_map(bank, map_idx)
        assert self.project_path is not None
        assert self.project is not None
        label = self.project.headers[bank][map_idx].label
        assert label is not None
        self.open_history.add(
            OpenHistoryItem(
                project_path=str(self.project_path.absolute()),
                bank=bank,
                map_idx=map_idx,
                label=label,
            )
        )
        self.build_open_recent_menu(self.file_menu_open_recent_menu)

    def open_footer(self, label: str, prompt_saving: bool = True):
        """Opens a new footer and assigns it to the current header."""
        if self.project is None or self.header is None:
            return
        if prompt_saving and (self.prompt_save_footer() or self.prompt_save_tilesets()):
            return
        self.map_widget.undo_stack.clear()
        self.footer_widget.undo_stack.clear()
        self.footer, footer_idx, serialized_smart_shapes = self.project.load_footer(
            label, map_blocks_to_ndarray=True, border_blocks_to_ndarray=True
        )
        self.footer_label = label
        # Associate this header with the new footer
        self.set_footer(self.footer_label, footer_idx)
        self.smart_shapes = {
            name: SmartShape.from_serialized(
                serialized_smart_shapes,
            )
            for name, serialized_smart_shapes in serialized_smart_shapes.items()
        }

        self.open_tilesets(
            self.get_tileset_label(True),
            self.get_tileset_label(False),
            prompt_saving=False,
        )  # Do not prompt saving the same files twice

    def open_tilesets(
        self,
        label_primary: str | None = None,
        label_secondary: str | None = None,
        prompt_saving: bool = True,
    ):
        """Opens and assigns tilesets to the footer."""
        if self.project is None or self.header is None or self.footer is None:
            return
        if prompt_saving and self.prompt_save_tilesets():
            return
        self.tileset_widget.undo_stack.clear()
        # Check if the tilesets need to be saved
        if label_primary is not None:
            # If the footer is assigned a null reference, do not render
            self.tileset_primary = self.project.load_tileset(True, label_primary)
            self.tileset_primary_label = label_primary
            self.set_tileset(label_primary, True)
        if label_secondary is not None:
            self.tileset_secondary = self.project.load_tileset(False, label_secondary)
            self.tileset_secondary_label = label_secondary
            self.set_tileset(label_secondary, False)

        if label_primary is not None or label_secondary is not None:
            self.open_gfxs(
                self.get_tileset_gfx_label(True), self.get_tileset_gfx_label(False)
            )

    def open_gfxs(
        self, label_primary: str | None = None, label_secondary: str | None = None
    ):
        """Opens and assigns new gfxs to the primary and secondary tilesets."""
        if not self.tilesets_loaded:
            return
        # Assign the gfxs to the tilesets
        if label_primary is not None:
            self.set_tileset_gfx(label_primary, True)
        if label_secondary is not None:
            self.set_tileset_gfx(label_secondary, False)
        if label_primary is not None or label_secondary is not None:
            # Load the gfx and render tiles
            self.load_blocks()
            self.update_gui()

    def load_blocks(self):
        """Updates blocks and their tiles."""
        assert self.project is not None
        self.tiles = render.get_tiles(
            self.tileset_primary, self.tileset_secondary, self.project
        )
        self.block_images = render.get_blocks(
            self.tileset_primary, self.tileset_secondary, self.tiles, self.project
        )

    def save_tilesets(self):
        """Saves the current tilesets."""
        if (
            self.project is None
            or self.header is None
            or self.footer is None
            or self.tileset_primary is None
            or self.tileset_secondary is None
        ):
            return
        self.project.save_tileset(
            True, self.tileset_primary, self.tileset_primary_label
        )
        self.project.save_tileset(
            False, self.tileset_secondary, self.tileset_secondary_label
        )
        self.tileset_widget.undo_stack.setClean()

    def save_footer(self):
        """Saves the current map footer."""
        if not self.footer_loaded:
            return
        assert self.project is not None, 'Project is not loaded'
        self.project.save_footer(
            self.footer,
            self.footer_label,
            {
                name: smart_shape.serialize()
                for name, smart_shape in self.smart_shapes.items()
            },
            map_blocks_to_list=True,
            border_blocks_to_list=True,
        )
        self.map_widget.undo_stack.setClean()
        self.footer_widget.undo_stack.setClean()

    def save_header(self):
        """Saves the current map header."""
        if not self.header_loaded:
            return
        assert self.project is not None, 'Project is not loaded'
        assert self.header_bank is not None, 'Header bank is not loaded'
        assert self.header_map_idx is not None, 'Header map index is not loaded'

        self.project.save_header(self.header, self.header_bank, self.header_map_idx)
        # Adapt history
        self.header_widget.undo_stack.setClean()
        self.event_widget.undo_stack.setClean()
        self.connection_widget.undo_stack.setClean()

    def update_gui(self):
        """Updates this gui and all child widgets."""
        self.tileset_widget.load_header()
        self.map_widget.load_header()
        self.footer_widget.load()
        self.header_widget.load()
        # It is important to place this after the map widget, since it reuses its tiling
        # self.event_widget.load_header()
        # self.connection_widget.load_header()

    def resource_tree_toggle_header_listing(self):
        """Toggles the listing method for the resource tree."""
        if not self.project_loaded:
            return

        self.settings.setValue(
            'resource_tree/header_listing',
            {
                HeaderSorting.BANK: HeaderSorting.NAMESPACE,
                HeaderSorting.NAMESPACE: HeaderSorting.BANK,
            }[
                self.settings.value(
                    'resource_tree/header_listing', HeaderSorting.NAMESPACE, str
                )
            ],  # type: ignore
        )
        self.resource_tree.load_headers()

    def toggle_grid(self):
        """Toggles the visibility of the grid."""
        self.settings.setValue(
            'grid_visible',
            not self.settings.value('grid_visible', False, bool),
        )
        if self.header_loaded:
            self.map_widget.update_grid()
            self.event_widget.update_grid()
            self.connection_widget.update_grid()

    @property
    def grid_visible(self) -> bool:
        """Returns whether the grid is visible."""
        return self.settings.value('grid_visible', False, bool)  # type: ignore

    def save_map_image(self):
        """Exports an image of the current map by issuing a prompt."""
        if (
            self.project is None
            or self.header is None
            or self.footer is None
            or self.tileset_primary is None
            or self.tileset_secondary is None
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Map Image',
            self.settings.value('map_image/recent', '.'),  # type: ignore
            'Portable Network Graphis (*.png)',
        )
        if len(path):
            self.settings.setValue('map_image/recent', os.path.dirname(path))
            image = QImage(
                self.map_widget.map_scene.sceneRect().size().toSize(),
                QImage.Format.Format_ARGB32,
            )
            painter = QPainter(image)
            self.map_widget.map_scene.render(painter)
            image.save(path)  # type: ignore
            painter.end()

    def event_widget_toggle_pictures(self):
        """Toggles if events are associated with pictures or not."""
        if self.project is None:
            return
        self.settings.setValue(
            'event_widget/show_pictures',
            not self.settings.value('event_widget/show_pictures', False, bool),
        )
        self.event_widget.load_header()

    def set_border_at(self, x: int, y: int, blocks: Tilemap):
        """Sets the blocks of the border and adds an action to the history."""
        if not self.footer_loaded:
            return
        border = self.get_borders()
        window = border[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[: window.shape[0], : window.shape[1]].copy()
        self.map_widget.undo_stack.push(SetBorder(self, x, y, blocks, window))

    def set_blocks_at(
        self,
        x: int,
        y: int,
        layers: MapLayers,
        blocks: Tilemap,
    ):
        """Sets the blocks on the header and adds an item to the history."""
        if not self.footer_loaded:
            return
        map_blocks = self.get_map_blocks()
        # Truncate blocks to fit the map
        window = map_blocks[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[: window.shape[0], : window.shape[1]].copy()
        self.map_widget.undo_stack.push(SetBlocks(self, x, y, layers, blocks, window))

    def shift_blocks(self, x: int, y: int, layers: MapLayers = [0, 1]):
        """Shifts the blocks in the current map footer."""
        if not self.footer_loaded:
            return
        blocks = self.get_map_blocks()[:, :, layers]  # h x w x layers
        map_width, map_height = self.get_map_dimensions()
        if x < 0:
            blocks = blocks[:, -x:]
            x = 0
        if y < 0:
            blocks = blocks[-y:, :]
            y = 0
        if x < map_width and y < map_height:
            self.set_blocks_at(x, y, layers, blocks)

    def flood_fill(self, x: int, y: int, layer: int, value: Tilemap):
        """Flood fills with origin (x, y) and a certain layer with a new value."""
        if not self.footer_loaded:
            return
        map_blocks = self.get_map_blocks()[..., layer]
        # Value 0 is not recognized by the connected component algorithm
        labeled: RGBAImage = label_image(map_blocks + 1, connectivity=1)  # type: ignore
        idx = np.where(labeled == labeled[y, x])
        self.map_widget.undo_stack.push(
            ReplaceBlocks(self, idx, layer, value, map_blocks[y, x])
        )

    def replace_blocks(self, x: int, y: int, layer: int, value: Tilemap) -> None:
        """Replaces all blocks that are like (x, y) in the layer by the new value."""
        map_blocks = self.get_map_blocks()[:, :, layer]
        idx = np.where(map_blocks == map_blocks[y, x])
        self.replace_blocks_at(
            idx=idx,
            layer=layer,
            value=value,
        )

    def replace_blocks_at(
        self,
        idx: tuple[NDArray[np.int_], ...],
        layer: int,
        value: Tilemap,
    ):
        """Replaces all blocks in the index by the new value."""
        if not self.footer_loaded:
            return
        map_blocks = self.get_map_blocks()[:, :, layer]
        value_old = map_blocks[idx].copy()  # type: ignore
        self.map_widget.undo_stack.push(
            ReplaceBlocks(self, idx, layer, value, value_old)  # type: ignore
        )

    def smart_shape_set_blocks_at(
        self,
        smart_shape_name: str,
        x: int,
        y: int,
        blocks: Tilemap,
    ):
        """Sets the blocks of a smart shape and adds an action to the history.

        Args:
            smart_shape_name (str): For which smart shape to set.
            x (int): The x coordinate.
            y (int): The y coordinate.
            layers (MapLayers): Which layers to set.
            blocks (RGBAImage): The blocks to set.
        """
        if not self.footer_loaded:
            return
        map_blocks = self.smart_shapes[smart_shape_name].buffer
        # Truncate the blocks to fit the map
        window = map_blocks[y : y + blocks.shape[0], x : x + blocks.shape[1]].copy()
        blocks = blocks[: window.shape[0], : window.shape[1]].copy()
        self.map_widget.undo_stack.push(
            SetSmartShapeBlocks(self, smart_shape_name, x, y, blocks, window)
        )

    def smart_shape_flood_fill(
        self,
        smart_shape_name: str,
        x: int,
        y: int,
        value: Tilemap,
        layer: int = 0,
    ):
        """Flood fills with origin (x, y) and a certain layer with a new value."""
        if not self.footer_loaded:
            return
        smart_shape_map_blocks = self.smart_shapes[smart_shape_name].buffer[..., layer]
        # Value 0 is not recognized by the connected component algorithm
        labeled: Tilemap = label_image(smart_shape_map_blocks + 1, connectivity=1)  # type: ignore
        idx = np.where(labeled == labeled[y, x])
        self.map_widget.undo_stack.push(
            SmartShapeReplaceBlocks(
                self, smart_shape_name, idx, layer, value, smart_shape_map_blocks[y, x]
            )
        )

    def smart_shape_replace_blocks(
        self,
        smart_shape_name: str,
        x: int,
        y: int,
        layer: int,
        value: Tilemap,
    ):
        """Replaces all blocks that are like (x, y) in the layer by the new value."""
        if not self.footer_loaded:
            return
        smart_shape_map_blocks = self.smart_shapes[smart_shape_name].buffer[..., layer]
        value_old = smart_shape_map_blocks[y, x].copy()
        idx = np.where(smart_shape_map_blocks == value_old)
        assert value_old.shape == value.shape
        self.map_widget.undo_stack.push(
            SmartShapeReplaceBlocks(
                self, smart_shape_name, idx, layer, value, value_old
            )
        )

    def smart_shape_clear(
        self,
        smart_shape_name: str,
        value: int = 0,
    ):
        """Clears the smart shape map."""
        if not self.footer_loaded:
            return
        smart_shape = self.smart_shapes[smart_shape_name]
        self.map_widget.undo_stack.push(
            SetSmartShapeBlocks(
                self,
                smart_shape_name,
                0,
                0,
                np.full_like(smart_shape.buffer, value),
                smart_shape.buffer.copy(),
            )
        )

    def resize_map(self, height_new: int, width_new: int):
        """Changes the map dimensions."""
        if not self.footer_loaded:
            return
        blocks = self.get_map_blocks()
        height, width = self.get_map_dimensions()
        if height != height_new or width != width_new:
            self.map_widget.undo_stack.push(
                ResizeMap(self, height_new, width_new, blocks)
            )

    def resize_border(self, height_new: int, width_new: int):
        """Changes the border dimensions."""
        if not self.header_loaded:
            return
        blocks = self.get_borders()
        width, height = self.get_border_dimensions()
        if height != height_new or width != width_new:
            self.map_widget.undo_stack.push(
                ResizeBorder(self, height_new, width_new, blocks)
            )

    def change_tileset(self, label: str, primary: bool):
        """Changes the current tileset by performing a command."""
        if not self.footer_loaded:
            return
        label_old = self.get_tileset_label(primary)
        self.map_widget.undo_stack.push(AssignTileset(self, primary, label, label_old))

    def change_footer(self, label: str):
        """Changes the current footer by performing a command on the header."""
        if not self.header_loaded:
            return
        self.header_widget.undo_stack.push(AssignFooter(self, label, self.footer_label))

    def change_gfx(self, label: str, primary: bool):
        """Changes the current gfx by performing a command."""
        if not self.tilesets_loaded:
            return
        label_old = self.get_tileset_gfx_label(primary)
        self.tileset_widget.undo_stack.push(AssignGfx(self, primary, label, label_old))

    def prompt_saving(self, parent: QWidget, text: str, informative_text: str) -> int:
        """Displays a prompt to ask the user if a certain file should be saved."""
        message_box = QMessageBox(parent)
        message_box.setWindowTitle(text)
        message_box.setText(informative_text)
        message_box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel
        )
        message_box.setDefaultButton(QMessageBox.StandardButton.Save)
        return message_box.exec_()  # type: ignore


def main():
    """Main entry point that runs the ui."""
    app = QApplication(sys.argv)
    app.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.Round)
    ex = PymapGui()
    ex.show()
    sys.exit(app.exec_())
