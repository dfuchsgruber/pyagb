"""Graphics scene with a transparent background for the map editor."""

from .transparent_tile import get_transparent_background
from .view import QGraphicsViewWithTransparentBackground

__all__ = ['QGraphicsViewWithTransparentBackground', 'get_transparent_background']
