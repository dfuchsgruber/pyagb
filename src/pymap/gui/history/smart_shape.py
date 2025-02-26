"""History actions for smart shapes."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import IntArray
from pymap.gui.smart_shape.smart_shape import SmartShape

if TYPE_CHECKING:
    from pymap.gui.main import PymapGui
    from pymap.gui.map.tabs.smart_shapes.edit_dialog.edit_dialog import (
        EditSmartShapeDialog,
    )
    from pymap.gui.map.tabs.smart_shapes.smart_shapes import SmartShapesTab


class AddOrRemoveSmartShape(QUndoCommand):
    """Adds a new smart shape."""

    def __init__(
        self,
        smart_shapes_tab: SmartShapesTab,
        name: str,
        smart_shape: SmartShape,
        add: bool,
    ):
        """Initializes the gfx assignment.

        Args:
            smart_shapes_tab (PymapGui): reference to the main gui
            name (str): Name of the smart shape
            smart_shape (SerializedSmartShape): The smart shape to add
            add (bool): Whether to add or remove the smart shape
        """
        super().__init__()
        self.smart_shapes_tab = smart_shapes_tab
        self.name = name
        self.smart_shape = smart_shape
        self.add = add

    def _add(self):
        """Helper for adding a smart shape."""
        assert (
            self.name not in self.smart_shapes_tab.map_widget.main_gui.smart_shapes
        ), 'Smart shape already exists.'
        self.smart_shapes_tab.map_widget.main_gui.smart_shapes[self.name] = deepcopy(
            self.smart_shape
        )
        self.smart_shapes_tab.update_smart_shapes(load_index=self.name)

    def _remove(self):
        """Helper for removing a smart shape."""
        assert (
            self.name in self.smart_shapes_tab.map_widget.main_gui.smart_shapes
        ), 'Smart shape does not exist.'
        del self.smart_shapes_tab.map_widget.main_gui.smart_shapes[self.name]
        self.smart_shapes_tab.update_smart_shapes()

    def redo(self):
        """Performs the smart shape assignment."""
        if self.add:
            self._add()
        else:
            self._remove()

    def undo(self):
        """Undoes the smart shape assignment."""
        if self.add:
            self._remove()
        else:
            self._add()


class SetSmartShapeTemplateBlocks(QUndoCommand):
    """Sets blocks in the template for a smart shape."""

    def __init__(
        self,
        edit_dialog: EditSmartShapeDialog,
        name: str,
        x: int,
        y: int,
        blocks_new: IntArray,
        blocks_old: IntArray,
    ):
        """Initializes the smart shape template blocks.

        Args:
            edit_dialog (SmartShapesTab): The smart shapes tab
            name (str): The name of the smart shape
            x (int): The x coordinate
            y (int): The y coordinate
            blocks_new (IntArray): The new blocks
            blocks_old (IntArray): The old blocks
        """
        super().__init__()
        self.edit_dialog = edit_dialog
        self.name = name
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old
        self.x = x
        self.y = y

    def _set_blocks(self, blocks: IntArray):
        """Helper for setting the blocks."""
        assert self.edit_dialog.smart_shapes_tab.map_widget.main_gui.project is not None
        smart_shape = (
            self.edit_dialog.smart_shapes_tab.map_widget.main_gui.smart_shapes[
                self.name
            ]
        )
        smart_shape.blocks[
            self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1]
        ] = blocks.copy()
        self.edit_dialog.update_shape_with_blocks(self.x, self.y, blocks)

    def redo(self):
        """Performs the setting of blocks."""
        self._set_blocks(self.blocks_new)

    def undo(self):
        """Undoes setting of blocks."""
        self._set_blocks(self.blocks_old)


class SetSmartShapeBlocks(QUndoCommand):
    """Set meta-blocks for a smart shape."""

    def __init__(
        self,
        main_gui: PymapGui,
        smart_shape_name: str,
        x: int,
        y: int,
        blocks_new: IntArray,
        blocks_old: IntArray,
    ):
        """Initializes the set meta blocks action.

        Args:
            main_gui (PymapGui): reference to the main gui
            smart_shape_name (str): The name of the smart shape
            x (int): at which x coordinate
            y (int): at which y coordinate
            blocks_new (IntArray): Which blocks to set
            blocks_old (IntArray): The old blocks
        """
        super().__init__()
        self.main_gui = main_gui
        self.x = x
        self.y = y
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old
        self.smart_shape_name = smart_shape_name

    def _set_blocks(self, blocks: IntArray):
        # Helper method for setting a set of blocks
        blocks = self.main_gui.smart_shapes[self.smart_shape_name].blocks
        blocks[
            self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], ...
        ] = blocks[:, :, ...]
        if (
            self.main_gui.map_widget.smart_shapes_tab.current_smart_shape_name
            == self.smart_shape_name
        ):
            self.main_gui.map_widget.update_map_with_smart_shape_blocks_at(
                self.x, self.y, blocks
            )

    def redo(self):
        """Performs the setting of blocks."""
        self._set_blocks(self.blocks_new)

    def undo(self):
        """Undos setting of blocks."""
        self._set_blocks(self.blocks_old)
