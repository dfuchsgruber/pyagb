"""Undo-Redo statements for the history of the GUI."""

from typing import Sequence

from agb.model.type import ModelValue

from pymap.configuration import AttributePathType

UndoRedoStatements = Sequence[str]


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
