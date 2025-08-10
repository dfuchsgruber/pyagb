"""Default templates for smart shapes."""

from .template import SmartShapeTemplate


def get_default_smart_shape_templates() -> dict[str, SmartShapeTemplate]:
    """Return the default templates."""
    from .polygon.polygon import SmartShapeTemplatePolygon

    return {'Polygon': SmartShapeTemplatePolygon()}
