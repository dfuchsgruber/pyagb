"""Items for gfxs in the resource tree."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu

from pymap.gui.icon.icon import Icon, icon_paths

from .item import ResourceParameterTreeItem

if TYPE_CHECKING:
    from pymap.gui.resource_tree.resource_tree import ResourceParameterTree


class ResourceParameterTreeItemGfxRoot(ResourceParameterTreeItem):
    """Root item for gfx roots."""

    def __init__(self, *args: Any, primary: bool, **kwargs: Any):
        """Initializes the gfx root item."""
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
        action = menu.addAction('Import Gfx')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.IMPORT]))
        action.triggered.connect(partial(tree.import_gfx, primary=self.primary))
        return menu


class ResourceParameterTreeItemGfx(ResourceParameterTreeItem):
    """Root item for footers."""

    def __init__(self, *args: Any, label: str, primary: bool, **kwargs: Any):
        """Initializes the header item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.label = label
        self.primary = primary

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Creates the context menu for the header item.

        Args:
            tree (ResourceParameterTree): The resource tree.

        Returns:
            QMenu | None: The context menu.
        """
        if tree.main_gui.project is None:
            return
        menu = QMenu()
        action = menu.addAction('Assign to Tileset')  # type: ignore
        action.triggered.connect(
            partial(tree.main_gui.change_gfx, label=self.label, primary=self.primary)
        )
        action = menu.addAction('Remove')  # type: ignore
        action.triggered.connect(
            partial(tree.remove_gfx, primary=self.primary, label=self.label)
        )
        action.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        action = menu.addAction('Relabel')  # type: ignore
        action.triggered.connect(
            partial(tree.refactor_gfx, primary=self.primary, label_old=self.label)
        )
        action.setIcon(QIcon(icon_paths[Icon.RENAME]))
        return menu
