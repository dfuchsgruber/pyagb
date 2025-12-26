"""Default templates for smart shapes."""

from .template import SmartShapeTemplate


def get_default_smart_shape_templates() -> dict[str, SmartShapeTemplate]:
    """Return the default templates."""
    from .polygon.polygon import SmartShapeTemplatePolygon
    from .polygon_double_height.polygon_double_height import (
        SmartShapeTemplatePolygonDoubleHeight,
    )

    return {
        'Polygon': SmartShapeTemplatePolygon(),
        'PolygonDoubleHeight': SmartShapeTemplatePolygonDoubleHeight(),
    }
