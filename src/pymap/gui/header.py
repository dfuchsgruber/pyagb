"""Tree widget for editing the map header properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QHeaderView, QWidget
from typing_extensions import ParamSpec

from pymap.gui import properties
from pymap.gui.history.statement import model_value_difference_to_undo_redo_statements

from .history import ChangeHeaderProperty

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


class HeaderWidget(ParameterTree):
    """Class for meta-properties of the map footer."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the header widget.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        self.main_gui = main_gui
        self.main_gui = main_gui
        self.root = None
        self.undo_stack = QUndoStack()
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.header().restoreState(  # type: ignore
            self.main_gui.settings.value('header_widget/header_state', b'', type=bytes)  # type: ignore
        )
        self.header().sectionResized.connect(  # type: ignore
            lambda: self.main_gui.settings.setValue(  # type: ignore
                'header_widget/header_state',
                self.header().saveState(),  # type: ignore
            )
        )

        self.load_project()

    def load_project(self, *args: Any):
        """Update project related widgets."""
        self.load_header()

    def _load_header(self):
        """Loads a new header if it is not None."""
        if not self.main_gui.header_loaded:
            return
        assert self.main_gui.project is not None, 'Project is None'
        header_datatype = self.main_gui.project.config['pymap']['header']['datatype']
        assert self.main_gui.header_bank is not None, 'Header bank is None'
        assert self.main_gui.header_map_idx is not None, 'Header map index is None'
        header_label = self.main_gui.project.headers[self.main_gui.header_bank][
            self.main_gui.header_map_idx
        ][0]
        assert header_label is not None, 'Header label is None'
        self.root = properties.type_to_parameter(
            self.main_gui.project, header_datatype
        )(
            header_label,
            self.main_gui.project,
            header_datatype,
            self.main_gui.header,
            [],
            None,
        )
        self.addParameters(self.root)  # type: ignore
        self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def load_header(self):
        """Loads a new header."""
        self.clear()
        self.root = None
        self._load_header()

    def update(self):
        """Updates all values in the tree according to the current footer."""
        self.root.blockSignals(True)  # type: ignore
        self.root.update(self.main_gui.header)  # type: ignore
        self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Handles changes in the tree.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        assert self.root is not None
        self.undo_stack.push(
            ChangeHeaderProperty(
                self,
                *model_value_difference_to_undo_redo_statements(
                    self.main_gui.header, self.root
                ),
            )
        )
