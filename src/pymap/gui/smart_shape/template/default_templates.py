"""Default templates for smart shapes."""

from .polygon.polygon import SmartShapeTemplatePolygon
from .template import SmartShapeTemplate


def get_default_smart_shape_templates() -> dict[str, SmartShapeTemplate]:
    """Return the default templates."""
    return {'Polygon': SmartShapeTemplatePolygon()}
