"""Items for headers in the resource tree."""

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


class ResourceParameterTreeItemHeaderRoot(ResourceParameterTreeItem):
    """Root item for headers."""

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
        if len(tree.main_gui.project.unused_banks()) == 0:
            return  # Spawn no context menu for the header root if
            # there are no map banks to add
        action = menu.addAction('Add Bank')  # type: ignore
        action.triggered.connect(tree.create_bank)
        action.setIcon(QIcon(icon_paths[Icon.PLUS]))
        action = menu.addAction('Import Header')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.IMPORT]))
        action.triggered.connect(lambda _: self.import_header())  # type: ignore
        return menu


class ResourceParameterTreeItemHeader(ResourceParameterTreeItem):
    """Root item for headers."""

    def __init__(self, *args: Any, bank: str, map_idx: str, **kwargs: Any):
        """Initializes the header item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.bank = bank
        self.map_idx = map_idx

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
        label, _, namespace = tree.main_gui.project.headers[self.bank][self.map_idx]
        action = menu.addAction('Open')  # type: ignore
        action.triggered.connect(
            partial(tree.main_gui.open_header, self.bank, self.map_idx)
        )
        action = menu.addAction('Remove')  # type: ignore
        action.triggered.connect(
            partial(tree.remove_header, bank=self.bank, map_idx=self.map_idx)
        )
        action.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        action = menu.addAction('Relabel')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.RENAME]))
        action.triggered.connect(
            partial(
                tree.refactor_header,
                bank=self.bank,
                map_idx=self.map_idx,
                namespace=namespace,
            )
        )
        action = menu.addAction('Change Namespace')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.TAG]))
        action.triggered.connect(
            partial(
                tree.refactor_header, bank=self.bank, map_idx=self.map_idx, label=label
            )
        )
        return menu

    def double_clicked(self, tree: ResourceParameterTree) -> None:
        """Double clicked the item.

        Args:
            tree (ResourceParameterTree): The resource tree.
        """
        tree.main_gui.open_header(self.bank, self.map_idx)


class ResourceParameterTreeItemNamespace(ResourceParameterTreeItem):
    """Root item for namespaces."""

    def __init__(self, *args: Any, namespace: str, **kwargs: Any):
        """Initializes the namespace item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.namespace = namespace

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Creates the context menu for the namespace item.

        Args:
            tree (ResourceParameterTree): The resource tree.

        Returns:
            QMenu | None: The context menu.
        """
        if tree.main_gui.project is None:
            return
        menu = QMenu()
        action = menu.addAction('Add Header')  # type: ignore
        action.setIcon(QIcon(icon_paths[Icon.PLUS]))
        action.triggered.connect(
            partial(tree.create_header, namespace=str(self.namespace))
        )
        return menu


class ResourceParameterTreeItemBank(ResourceParameterTreeItem):
    """Root item for secondary gfx roots."""

    def __init__(self, *args: Any, bank: str, **kwargs: Any):
        """Initializes the bank item."""
        super().__init__(*args, **kwargs)  # type: ignore
        self.bank = bank

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Creates the context menu for the bank item.

        Args:
            tree (ResourceParameterTree): The resource tree.

        Returns:
            QMenu | None: The context menu.
        """
        assert tree.main_gui.project is not None
        menu = QMenu()
        bank = self.bank
        if len(tree.main_gui.project.unused_map_idx(bank)) > 0:
            action = menu.addAction('Add Header')  # type: ignore
            action.triggered.connect(partial(tree.create_header, bank=bank))
            action.setIcon(QIcon(icon_paths[Icon.PLUS]))
            action = menu.addAction('Import Header')  # type: ignore
            action.setIcon(QIcon(icon_paths[Icon.IMPORT]))
            action.triggered.connect(lambda _: tree.import_header(bank=bank))  # type: ignore
        action = menu.addAction('Remove')  # type: ignore
        action.triggered.connect(partial(tree.remove_bank, bank=bank))
        action.setIcon(QIcon(icon_paths[Icon.REMOVE]))
        return menu
