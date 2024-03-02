"""Items for headers in the resource tree."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMenu

from pymap.gui.icon.icon import Icon, icon_paths

from .item import ResourceParameterTreeItem

if TYPE_CHECKING:
    from pymap.gui.resource_tree.resource_tree import ResourceParameterTree


class ResourceParameterTreeItemFooterRoot(ResourceParameterTreeItem):
    """Root item for footer."""

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
        action = menu.addAction('Add Footer')  # type: ignore
        action.triggered.connect(tree.create_footer)
        action.setIcon(QIcon(icon_paths[Icon.PLUS]))
        action = menu.addAction('Import Footer')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.IMPORT]))
        action.triggered.connect(tree.import_footer)
        return menu


class ResourceParameterTreeItemFooter(ResourceParameterTreeItem):
    """Root item for footers."""

    def __init__(self, *args: Any, label: str, **kwargs: Any):
        """Initializes the header item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.label = label

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
        action = menu.addAction('Assign to Header')  # type: ignore
        action.triggered.connect(partial(tree.main_gui.change_footer, label=self.label))
        action = menu.addAction('Remove')  # type: ignore
        action.triggered.connect(partial(tree.remove_footer, footer=self.label))
        action.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        action = menu.addAction('Relabel')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.RENAME]))
        action.triggered.connect(partial(tree.refactor_footer, label_old=self.label))
        return menu
