"""Undo-Redo statements for the history of the GUI."""

from typing import Sequence, TypedDict

import numpy as np
from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelContext, ModelValue, ScalarModelValue
from pymap.configuration import AttributePathType

UndoRedoStatements = Sequence[str]


class ModelValueDifference(TypedDict):
    """Dict for the difference between model values."""

    old_value: ModelValue
    new_value: ModelValue


class ChangeProperty(QUndoCommand):
    """Class to change the property of some model value."""

    def __init__(
        self,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
        text: str = 'Change Property',
    ):
        """Initializes the property change.

        Args:
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
            text (str): The text for the action
        """
        super().__init__(text)
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


def model_value_difference(  # noqa: C901
    old: ModelValue, new: ModelValue
) -> dict[ModelContext, ModelValueDifference]:
    """Compares two model values and returns the differences.

    For now, we do not handle additions and deletions, those should not occur...

    Args:
        old (ModelValue): The old value
        new (ModelValue): The new value

    Returns:
        dict[ModelContext, ModelValueDifference]: The differences
    """
    if isinstance(old, np.ndarray):
        return model_value_difference(old.tolist(), new)
    elif isinstance(new, np.ndarray):
        return model_value_difference(old, new.tolist())
    elif isinstance(old, ScalarModelValue) and isinstance(new, ScalarModelValue):
        if str(old) != str(new):
            return {(): {'old_value': old, 'new_value': new}}
        else:
            return {}
    elif isinstance(old, dict) and isinstance(new, dict):
        # Recursively compare the values of the dictionaries
        diffs: dict[ModelContext, ModelValueDifference] = {}
        for key in old.keys() & new.keys():
            for path, diff in model_value_difference(old[key], new[key]).items():
                diffs[(key,) + tuple(path)] = diff
        return diffs
    elif isinstance(old, Sequence) and isinstance(new, Sequence):
        diffs: dict[ModelContext, ModelValueDifference] = {}
        for i, (old_value, new_value) in enumerate(zip(old, new)):
            for path, diff in model_value_difference(old_value, new_value).items():
                diffs[(i,) + tuple(path)] = diff
        return diffs
    else:
        raise NotImplementedError(
            f'Comparison of {type(old)} and {type(new)} not supported'
        )


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
    statements_redo: UndoRedoStatements = []
    statements_undo: UndoRedoStatements = []
    for path, diff in model_value_difference(old_value, new_value).items():
        statement_redo, statement_undo = path_to_statement(
            path, diff['old_value'], diff['new_value']
        )
        statements_redo.append(statement_redo)
        statements_undo.append(statement_undo)
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
