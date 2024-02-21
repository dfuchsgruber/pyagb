"""Scene for the individual blocks."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec

_P = ParamSpec("_P")

if TYPE_CHECKING:
    from .tileset import TilesetWidget

def if_tileset_loaded(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to check if the tileset is loaded."""
    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        self = args[0]
        assert isinstance(self, TilesetChildWidgetMixin)
        if not self.tileset_widget.tileset_loaded:
            return
        return func(*args, **kwargs)
    return wrapper


class TilesetChildWidgetMixin:
    """Mixin for widgets that are children of the tileset widget."""

    def __init__(self, tileset_widget: TilesetWidget):
        """Initializes the mixin.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
        """
        self.tileset_widget = tileset_widget

