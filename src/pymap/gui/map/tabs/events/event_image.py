"""Class for the image of a map event."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pymap.gui.types import RGBAImage


class EventImage(NamedTuple):
    """An image for a map event."""

    image: RGBAImage
    x_offset: int
    y_offset: int
