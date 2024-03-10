"""Scene for events."""


from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .event_widget import EventWidget


def if_header_loaded(func: Callable[_P, None]) -> Callable[_P, None]:
    """Decorator to check if the header is loaded."""

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        self = args[0]
        assert isinstance(self, EventChildWidgetMixin)
        if not self.event_widget.header_loaded:
            return
        return func(*args, **kwargs)

    return wrapper


class EventChildWidgetMixin:
    """Mixin for widgets that are children of the tileset widget."""

    def __init__(self, event_widget: EventWidget):
        """Initializes the mixin.

        Args:
            event_widget (EventWidget): The event widget.
        """
        self.event_widget = event_widget
