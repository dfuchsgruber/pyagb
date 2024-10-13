"""Items for the resource tree."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMenu, QTreeWidgetItem

if TYPE_CHECKING:
    from pymap.gui.resource_tree.resource_tree import ResourceParameterTree


class Context(StrEnum):
    """Enum for the context of the resource tree."""

    HEADER_ROOT = 'header_root'
    BANK = 'bank'

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


class ResourceParameterTreeItem(QTreeWidgetItem):
    """Root item for the resource tree."""

    def context_menu(self, tree: ResourceParameterTree) -> QMenu | None:
        """Opens the context menu for the item."""
        return None

    def double_clicked(self, tree: ResourceParameterTree) -> None:
        """Double clicked the item."""
        pass
