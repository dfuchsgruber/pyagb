"""Tree widget for editing the map footer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepdiff import DeepDiff  # type: ignore
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QHeaderView, QWidget
from typing_extensions import ParamSpec

from pymap.gui import properties
from pymap.gui.history import UndoRedoStatements

from .history import ChangeFooterProperty

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


class FooterWidget(ParameterTree):
    """Class for meta-properties of the map footer."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the footer widget.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        self.main_gui = main_gui
        self.root = None
        self.undo_stack = QUndoStack()
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.load_project()

    def load_project(self, *args: Any):
        """Update project related widgets."""
        self.load_footer()

    def load_footer(self):
        """Loads a new footer."""
        self.clear()
        if (
            self.main_gui.project is not None
            and self.main_gui.header is not None
            and self.main_gui.footer is not None
        ):
            assert self.main_gui.footer_label is not None, 'Footer label is None'
            footer_datatype = self.main_gui.project.config['pymap']['footer'][
                'datatype'
            ]
            self.root = properties.type_to_parameter(
                self.main_gui.project, footer_datatype
            )(
                self.main_gui.footer_label,
                self.main_gui.project,
                footer_datatype,
                self.main_gui.footer,
                [],
                None,
            )
            self.addParameters(self.root)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore
        else:
            self.root = None

    def update(self):
        """Updates all values in the tree according to the current footer."""
        if self.root is not None:
            self.root.blockSignals(True)  # type: ignore
            self.root.update(self.main_gui.footer)
            self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Handles changes in the tree.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        if self.root is not None:
            diffs = DeepDiff(self.main_gui.footer, self.root.model_value)
            root = self.main_gui.footer  # type: ignore # noqa: F841
            statements_redo: UndoRedoStatements = []
            statements_undo: UndoRedoStatements = []
            for change in ('type_changes', 'values_changed'):
                if change in diffs:
                    for path in diffs[change]:  # type: ignore
                        value_new = diffs[change][path]['new_value']  # type: ignore
                        value_old = diffs[change][path]['old_value']  # type: ignore
                        statements_redo.append(f"{path} = '{value_new}'")
                        statements_undo.append(f"{path} = '{value_old}'")
            self.undo_stack.push(
                ChangeFooterProperty(self, statements_redo, statements_undo)
            )
