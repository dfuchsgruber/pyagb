"""Properties tree for any ModelValue."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from agb.model.type import ModelContext, ModelValue
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import (
    QHeaderView,
    QWidget,
)
from typing_extensions import ParamSpec

from pymap.gui import properties
from pymap.gui.history import (
    model_value_difference_to_undo_redo_statements,
)
from pymap.gui.history.statement import UndoRedoStatements

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from ..main.gui import PymapGui


class ModelValueNotAvailableError(Exception):
    """Model value not available error."""


class PropertiesTree(ParameterTree):
    """Tree to display event properties."""

    def __init__(
        self,
        name: str,
        main_gui: PymapGui,
        parent: QWidget | None = None,
    ):
        """Initializes the event properties.

        Args:
            name (str): The name of the parameter.
            datatype (str): The datatype.
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): Parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        self.main_gui = main_gui
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.header().restoreState(  # type: ignore
            self.main_gui.settings.value(
                f'{name}/header_state',
                b'',
                type=bytes,
            )
        )
        self.header().sectionResized.connect(  # type: ignore
            lambda: self.main_gui.settings.setValue(  # type: ignore
                f'{name}/header_state',
                self.header().saveState(),  # type: ignore
            )
        )
        self.root = None

    @property
    @abstractmethod
    def model_value(self) -> ModelValue:
        """Returns the currently displayed event.

        Returns:
            ModelValue: The currently displayed event.

        Raises:
            ModelValueNotAvailableError: If the model value is not available.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datatype(self) -> str:
        """Returns the datatype of the event.

        Returns:
            str: The datatype of the event.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_context(self) -> ModelContext:
        """Returns the context of the event."""
        raise NotImplementedError

    @abstractmethod
    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ) -> None:
        """Signal handler for when the value changes.

        Args:
            statements_redo (UndoRedoStatements): The redo statements.
            statements_undo (UndoRedoStatements): The undo statements.
        """
        raise NotImplementedError

    def load(self):
        """Loads the currently displayed event."""
        self.clear()
        try:
            value = self.model_value
        except ModelValueNotAvailableError:
            self.root = None
            return
        else:
            assert self.main_gui.project is not None, 'Project is None'
            self.root = properties.type_to_parameter(
                self.main_gui.project, self.datatype
            )(
                '.',
                self.main_gui.project,
                self.datatype,
                value,
                self.model_context,
                None,
            )
            self.addParameters(self.root, showTop=False)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def update(self):
        """Updates all values in the tree according to the current event."""
        assert self.root is not None, 'Root is None'
        self.set_value(self.model_value)

    def set_value(self, value: ModelValue):
        """Replaces the entrie properties of the current block if one is selected."""
        self.root.blockSignals(True)  # type: ignore
        self.root.update(value)  # type: ignore
        self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Signal handler for when the tree changes.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        assert self.root is not None, 'Root is None'
        root = self.model_value
        self.value_changed(
            *model_value_difference_to_undo_redo_statements(
                root, self.root.model_value
            ),
        )
