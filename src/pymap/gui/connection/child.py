"""Children of the connection widget."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .connection_widget import ConnectionWidget


def if_connection_loaded(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to check if a connection is loaded."""

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        self = args[0]
        assert isinstance(self, ConnectionChildWidgetMixin)
        if not self.connection_widget.connection_loaded:
            return
        return func(*args, **kwargs)

    return wrapper


class ConnectionChildWidgetMixin:
    """Mixin for widgets that are children of the connection widget."""

    def __init__(self, connection_widget: ConnectionWidget):
        """Initializes the mixin.

        Args:
            connection_widget (TilesetWidget): The tileset widget.
        """
        self.connection_widget = connection_widget
