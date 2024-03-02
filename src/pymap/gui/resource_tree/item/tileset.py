"""Items for tilesets in the resource tree."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu

from pymap.gui.icon.icon import Icon, icon_paths
from pymap.gui.resource_tree.resource_tree import ResourceParameterTree

from .item import ResourceParameterTreeItem

if TYPE_CHECKING:
    from pymap.gui.resource_tree.resource_tree import ResourceParameterTree


class ResourceParameterTreeItemTilesetRoot(ResourceParameterTreeItem):
    """Root item for tilesets."""

    def __init__(self, *args: Any, primary: bool, **kwargs: Any):
        """Initializes the tileset root item."""
        super().__init__(*args, **kwargs)
        self.primary = primary

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Opens the context menu for the item.

        Args:
            tree (ResourceParameterTree): The resource tree.

        Returns:
            QMenu | None: The context menu.
        """
        if tree.main_gui.project is None:
            return
        menu = QMenu()
        action = menu.addAction('Add Tileset')  # type: ignore
        action.triggered.connect(partial(tree.create_tileset, primary=self.primary))
        action.setIcon(QIcon(icon_paths[Icon.PLUS]))
        action = menu.addAction('Import Tileset')  # type: ignore
        action.triggered.connect(partial(tree.import_tileset, primary=self.primary))
        action.setIcon(QIcon(icon_paths[Icon.IMPORT]))


class ResourceParameterTreeItemTileset(ResourceParameterTreeItem):
    """Item for tilesets."""

    def __init__(self, *args: Any, primary: bool, label: str, **kwargs: Any):
        """Initializes the tileset item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.primary = primary
        self.label = label

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Creates the context menu for the tileset item.

        Args:
            tree (ResourceParameterTree): The resource tree.

        Returns:
            QMenu | None: The context menu.
        """
        if tree.main_gui.project is None:
            return
        menu = QMenu()
        action = menu.addAction('Assign to Footer')  # type: ignore
        action.triggered.connect(
            partial(tree.main_gui.change_tileset, self.label, primary=self.primary)
        )
        action = menu.addAction('Remove')  # type: ignore
        action.triggered.connect(
            partial(tree.remove_tileset, primary=self.primary, label=self.label)
        )
        action.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        action = menu.addAction('Relabel')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.RENAME]))
        action.triggered.connect(
            partial(tree.refactor_tileset, primary=self.primary, label_old=self.label)
        )
        action = menu.addAction('Duplicate')  # type: ignore
        action.triggered.connect(
            partial(tree.duplicate_tileset, src_label=self.label, primary=self.primary)
        )
        return menu
