"""Child of the Map tab."""


from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .map_widget import MapWidget


def if_header_loaded(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to check if the header is loaded."""

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        self = args[0]
        assert isinstance(self, MapChildMixin)
        if not self.map_widget.header_loaded:
            return
        return func(*args, **kwargs)

    return wrapper


class MapChildMixin:
    """Mixin for widgets that are children of the map widget."""

    def __init__(self, map_widget: MapWidget):
        """Initializes the mixin.

        Args:
            map_widget (EventWidget): The event widget.
        """
        self.map_widget = map_widget
