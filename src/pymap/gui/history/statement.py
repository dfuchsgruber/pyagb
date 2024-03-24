"""Undo-Redo statements for the history of the GUI."""

from typing import Sequence

from agb.model.type import ModelValue
from deepdiff import DeepDiff  # type: ignore
from PySide6.QtGui import QUndoCommand

from pymap.configuration import AttributePathType

UndoRedoStatements = Sequence[str]


class ChangeProperty(QUndoCommand):
    """Class to change the property of some model value."""

    def __init__(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ):
        """Initializes the property change.

        Args:
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__()
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        raise NotImplementedError

    def redo(self) -> None:
        """Redoes the property change."""
        root = self.get_root()  # type: ignore   # noqa: F841
        for statement in self.statements_redo:
            exec(statement)

    def undo(self) -> None:
        """Undoes the property change."""
        root = self.get_root()  # type: ignore   # noqa: F841
        for statement in self.statements_undo:
            exec(statement)


def model_value_difference_to_undo_redo_statements(
    old_value: ModelValue, new_value: ModelValue
) -> tuple[UndoRedoStatements, UndoRedoStatements]:
    """Transforms the difference between two model values into undo-redo statements.

    Args:
        old_value (ModelValue): The old value
        new_value (ModelValue): The new value

    Returns:
        tuple[UndoRedoStatements, UndoRedoStatements]: The undo and redo statements
    """
    diffs = DeepDiff(old_value, new_value)
    statements_redo: UndoRedoStatements = []
    statements_undo: UndoRedoStatements = []

    for change in ('values_changed',):
        if change in diffs:
            for path in diffs[change]:  # type: ignore
                value_new = diffs[change][path]['new_value']  # type: ignore
                value_old = diffs[change][path]['old_value']  # type: ignore
                statements_redo.append(f"{path} = '{value_new}'")
                statements_undo.append(f"{path} = '{value_old}'")
    return statements_redo, statements_undo


def path_to_statement(
    path: AttributePathType, old_value: ModelValue, new_value: ModelValue
) -> UndoRedoStatements:
    """Transforms a path to a property into a redoable statement.

    All statements are evaluated relative to a local scope that includes
    a `root` variable that points to the root object of the property.
    """
    path = ''.join(map(lambda member: f'[{repr(member)}]', path))
    return [
        f'root{path} = {repr(str(new_value))}',
        f'root{path} = {repr(str(old_value))}',
    ]
