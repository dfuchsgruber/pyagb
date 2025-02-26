"""Polygon auto-shape template."""

import importlib.resources as resources

from ..template import SmartShapeTemplate


class SmartShapeTemplatePolygon(SmartShapeTemplate):
    """Polygon smart shape template."""

    def __init__(self):
        """Initialize the template."""
        super().__init__(
            str(
                resources.files('pymap.gui.smart_shape.template.polygon').joinpath(
                    'template.png'
                )
            )
        )
