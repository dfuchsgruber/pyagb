"""Mixin for the children of the main gui."""

from __future__ import annotations

from functools import partial, wraps
from typing import TYPE_CHECKING, Callable

from typing_extensions import ParamSpec

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .gui import PymapGui


def _if_any_loaded(
    func: Callable[_P, None], condition: Callable[[PymapGui], bool]
) -> Callable[_P, None]:
    """Decorator to check if a condition on the main gui holds."""

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        self = args[0]
        assert isinstance(self, MainGuiChildWidgetMixin)
        if not condition(self.main_gui):
            return
        return func(*args, **kwargs)

    return wrapper


if_header_loaded = partial(_if_any_loaded, condition=lambda x: x.header_loaded)
if_footer_loaded = partial(_if_any_loaded, condition=lambda x: x.footer_loaded)
if_project_loaded = partial(_if_any_loaded, condition=lambda x: x.project_loaded)


class MainGuiChildWidgetMixin:
    """Mixin for widgets that are children of the main gui."""

    def __init__(self, main_gui: PymapGui):
        """Initializes the mixin.

        Args:
            main_gui (PymapGui): The main gui.
        """
        self.main_gui = main_gui
