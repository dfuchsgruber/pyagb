"""Actions for changing values of the data model."""

from .blocks import ReplaceBlocks, ResizeBorder, ResizeMap, SetBlocks, SetBorder
from .connection import AppendConnection, ChangeConnectionProperty, RemoveConnection
from .event import AppendEvent, ChangeEventProperty, RemoveEvent
from .footer import AssignTileset, ChangeFooterProperty
from .header import AssignFooter, ChangeHeaderProperty
from .smart_shape import AddOrRemoveSmartShape
from .statement import (
    UndoRedoStatements,
    model_value_difference_to_undo_redo_statements,
    path_to_statement,
)
from .tileset import (
    AssignGfx,
    ChangeBlockProperty,
    ChangeTilesetProperty,
    SetPalette,
    SetTiles,
)

__all__ = [
    'AppendConnection',
    'AppendEvent',
    'AssignFooter',
    'AssignGfx',
    'AssignTileset',
    'ChangeBlockProperty',
    'ChangeConnectionProperty',
    'ChangeFooterProperty',
    'ChangeHeaderProperty',
    'ChangeEventProperty',
    'ChangeTilesetProperty',
    'path_to_statement',
    'RemoveConnection',
    'RemoveEvent',
    'ReplaceBlocks',
    'ResizeBorder',
    'ResizeMap',
    'SetBlocks',
    'SetBorder',
    'SetPalette',
    'SetTiles',
    'UndoRedoStatements',
    'model_value_difference_to_undo_redo_statements',
    'AddOrRemoveSmartShape',
]
