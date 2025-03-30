"""Base class for smart shapes."""

from typing import TypedDict

import numpy as np

from pymap.gui.types import RGBAImage


class SerializedSmartShape(TypedDict):
    """A serialized smart shape."""

    template: str
    blocks: list[list[int]]  # height x width
    buffer: list[list[list[int]]]  # map_height x map_width


class SmartShape:
    """Base class for smart shape realizations.

    Every shape uses a template that specifies
        i) which smart-shape-meta-blocks can be used for mapping the shape to the map
        ii) which block is used for which part of the shape (`template_blocks`)

    Each smart shape also holds a buffer that maps the smart-shape-meta-blocks
    to the map.
    """

    def __init__(
        self,
        template: str,
        template_blocks: list[list[int]],
        buffer_blocks: list[list[list[int]]],
    ):
        """Initialize the smart shape."""
        self.template = template
        # The blocks are the template blocks mapped to the map
        self.blocks: RGBAImage = np.array(template_blocks, dtype=int)
        # The buffer is what is actually mapped to the map
        # It is transient, i.e. not serialized
        self.buffer: RGBAImage = np.array(buffer_blocks, dtype=int)

    def serialize(self) -> SerializedSmartShape:
        """Serialize the smart shape."""
        return {
            'template': self.template,
            'blocks': self.blocks.tolist(),  # type: ignore
            'buffer': self.buffer.tolist(),  # type: ignore
        }

    @classmethod
    def from_serialized(cls, serialized: SerializedSmartShape):
        """Create a smart shape from a serialized smart shape."""
        return cls(serialized['template'], serialized['blocks'], serialized['buffer'])
