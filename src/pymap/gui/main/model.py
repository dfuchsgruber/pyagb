"""Separate class to handel functionality for the ModelValues class."""


from abc import abstractmethod
from pathlib import Path

import numpy as np
from agb.model.type import ModelValue
from numpy.typing import NDArray
from PIL import Image

from pymap.configuration import PymapEventConfigType
from pymap.gui.properties import get_member_by_path, set_member_by_path
from pymap.project import Project


class PymapGuiModel:
    """Separate class to handel functionality for the model."""

    def __init__(self):
        """Initializes the model."""
        self.project: Project | None = None
        self.project_path: Path | None = None
        self.header: ModelValue | None = None
        self.header_bank: str | None = None
        self.header_map_idx: str | None = None
        self.footer: ModelValue | None = None
        self.footer_label = None
        self.tileset_primary = None
        self.tileset_primary_label = None
        self.tileset_secondary = None
        self.tileset_secondary_label = None
        self.blocks: list[Image.Image] | None = None
        self.tiles: list[list[Image.Image]] | None = None

    @property
    def project_loaded(self) -> bool:
        """Whether a project is currently loaded."""
        return self.project is not None

    @property
    def header_loaded(self) -> bool:
        """Returns whether a header is currently loaded."""
        return self.header is not None and self.project_loaded

    @property
    def footer_loaded(self) -> bool:
        """Returns whether a footer is currently loaded."""
        return self.footer is not None and self.header_loaded

    @property
    def tilesets_loaded(self) -> bool:
        """Returns whether the tilesets are currently loaded."""
        return self.tileset_primary is not None and self.tileset_secondary is not None

    @property
    @abstractmethod
    def show_borders(self) -> bool:
        """Whether borders should be shown.

        Returns:
            bool: Whether borders should be shown.
        """
        ...

    def get_map_dimensions(self) -> tuple[int, int]:
        """Gets the map dimensions.

        Returns:
            tuple[int, int]: The map dimensions (w, h).
        """
        assert self.project is not None
        map_width = get_member_by_path(
            self.footer,
            self.project.config['pymap']['footer']['map_width_path'],
        )
        assert isinstance(map_width, int), f'Expected int, got {type(map_width)}'

        map_height = get_member_by_path(
            self.footer,
            self.project.config['pymap']['footer']['map_height_path'],
        )
        assert isinstance(map_height, int), f'Expected int, got {type(map_height)}'
        return map_width, map_height

    def get_connections(self) -> list[ModelValue]:
        """Gets the connections of the current map.

        Returns:
            list[ModelValue]: The connections
        """
        assert self.project is not None, 'Project is None'
        connections = get_member_by_path(
            self.header,
            self.project.config['pymap']['header']['connections']['connections_path'],
        )
        assert isinstance(connections, list), f'Expected list, got {type(connections)}'
        return connections

    def get_events(self, event_type: PymapEventConfigType) -> list[ModelValue]:
        """Gets the events of the header.

        Args:
            event_type (PymapEventConfigType): The event type to get

        Returns:
            list[ModelValue]: The events
        """
        assert self.header is not None, 'Header is None'
        events = get_member_by_path(self.header, event_type['events_path'])
        assert isinstance(events, list), f'Expected list, got {type(events)}'
        return events

    def get_event(self, event_type: PymapEventConfigType, event_idx: int) -> ModelValue:
        """Gets an event from the header.

        Args:
            event_type (PymapEventConfigType): The event type to get
            event_idx (int): The index of the event to get

        Returns:
            ModelValue: The event
        """
        return self.get_events(event_type)[event_idx]

    def get_border_padding(self) -> tuple[int, int]:
        """Returns how many blocks are padded to the border of the map."""
        if self.show_borders:
            assert self.project is not None, 'Project is None'
            padding = tuple(self.project.config['pymap']['display']['border_padding'])
            assert len(padding) == 2, f'Expected 2, got {len(padding)}'
            return padding
        else:
            return 0, 0

    def get_borders(self) -> NDArray[np.int_]:
        """Gets the borders of the map.

        Returns:
            npt.NDArray[np.int_]: The borders
        """
        assert self.project is not None, 'Project is None'
        borders = get_member_by_path(
            self.footer,
            self.project.config['pymap']['footer']['border_path'],
        )
        assert isinstance(borders, np.ndarray), 'Borders are not numpy array'
        return borders

    def get_block(self, block_idx: int) -> NDArray[np.object_]:
        """Returns the block of the currently detected tileset.

        Args:
            block_idx (int): The index of the block.

        Returns:
            npt.NDArray[np.object_]: The block of shape [layer, h, w]
        """
        assert self.project is not None
        blocks = get_member_by_path(
            self.tileset_primary if block_idx < 0x280 else self.tileset_secondary,
            self.project.config['pymap'][
                'tileset_primary' if block_idx < 0x280 else 'tileset_secondary'
            ]['blocks_path'],
        )
        assert isinstance(blocks, list)
        return np.array(blocks[block_idx % 0x280]).reshape(3, 2, 2)

    def get_map_blocks(self) -> NDArray[np.int_]:
        """Gets the map blocks.

        Returns:
            npt.NDArray[np.int_]: The map blocks, shape [h, w, 2]
        """
        assert self.project is not None, 'Project is None'
        blocks = get_member_by_path(
            self.footer,
            self.project.config['pymap']['footer']['map_blocks_path'],
        )
        assert isinstance(blocks, np.ndarray), 'Blocks are not numpy array'
        return blocks

    def get_footer_label(self) -> str:
        """Gets the label of the current footer.

        Returns:
            str: The label of the footer.
        """
        assert self.project is not None, 'Project is None'
        assert self.header is not None, 'Header is None'
        label = get_member_by_path(
            self.header,
            self.project.config['pymap']['header']['footer_path'],
        )
        assert isinstance(label, str), f'Expected str, got {type(label)}'
        return label

    def get_tileset_label(self, primary: bool) -> str:
        """Gets the label of the current tileset from the footer.

        Args:
            primary (bool): Whether to get the primary or secondary tileset.

        Returns:
            str: The label of the tileset.
        """
        assert self.project is not None, 'Project is None'
        assert self.footer is not None, 'Footer is None'
        label = get_member_by_path(
            self.footer,
            self.project.config['pymap']['footer'][
                'tileset_primary_path' if primary else 'tileset_secondary_path'
            ],
        )
        assert isinstance(label, str), f'Expected str, got {type(label)}'
        return label

    def get_tileset_gfx_label(self, primary: bool) -> str:
        """Gets the label of the current tileset from the gfx.

        Args:
            primary (bool): Whether to get the primary or secondary tileset.

        Returns:
            str: The label of the tileset.
        """
        assert self.project is not None, 'Project is None'
        label = get_member_by_path(
            self.tileset_primary if primary else self.tileset_secondary,
            self.project.config['pymap'][
                'tileset_primary' if primary else 'tileset_secondary'
            ]['gfx_path'],
        )
        assert isinstance(label, str), f'Expected str, got {type(label)}'
        return label

    def get_tileset_behvaiours(self, primary: bool) -> list[ModelValue]:
        """Gets the behaviours of the current tileset.

        Args:
            primary (bool): Whether to get the primary or secondary tileset.

        Returns:
            list[ModelValue]: The behaviours
        """
        assert self.project is not None, 'Project is None'
        behaviours = get_member_by_path(
            self.tileset_primary if primary else self.tileset_secondary,
            self.project.config['pymap'][
                'tileset_primary' if primary else 'tileset_secondary'
            ]['behaviours_path'],
        )
        assert isinstance(behaviours, list)
        return behaviours

    def get_tileset_behaviour(self, block_idx: int) -> ModelValue:
        """Gets the behaviour of a block.

        Args:
            block_idx (int): The index of the block.

        Returns:
            ModelValue: The behaviour
        """
        behaviours = self.get_tileset_behvaiours(block_idx < 0x280)
        return behaviours[block_idx % 0x280]

    def get_tileset_palettes(self, primary: bool) -> list[ModelValue]:
        """Gets the palettes of the a tileset.

        Args:
            primary (bool): Whether to get the primary or secondary tileset.

        Returns:
            list[ModelValue]: The palettes
        """
        assert self.project is not None, 'Project is None'
        palettes = get_member_by_path(
            self.tileset_primary if primary else self.tileset_secondary,
            self.project.config['pymap'][
                'tileset_primary' if primary else 'tileset_secondary'
            ]['palettes_path'],
        )
        assert isinstance(palettes, list), f'Expected list, got {type(palettes)}'
        return palettes

    def get_tileset_palette(self, palette_idx: int) -> list[ModelValue]:
        """Gets the palette of a tileset.

        Args:
            palette_idx (int): The index of the palette.

        Returns:
            list[ModelValue]: The palette
        """
        palettes = self.get_tileset_palettes(palette_idx < 7)
        palette = palettes[palette_idx % 7]
        assert isinstance(palette, list), f'Expected list, got {type(palette)}'
        return palette

    def get_num_events(self, event_type: PymapEventConfigType) -> int:
        """Gets the number of events of the header.

        Args:
            event_type (PymapEventConfigType): The event type to get

        Returns:
            int: The number of events
        """
        num_events = get_member_by_path(
            self.header,
            event_type['size_path'],
        )
        assert isinstance(num_events, int), f'Expected int, got {type(num_events)}'
        return num_events

    def set_footer(self, label: str, footer_idx: int):
        """Sets the footer of the header.

        Args:
            label (str): The label of the footer.
            footer_idx (int): The index of the footer.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.header, label, self.project.config['pymap']['header']['footer_path']
        )
        set_member_by_path(
            self.header,
            footer_idx,
            self.project.config['pymap']['header']['footer_idx_path'],
        )

    def set_tileset(self, label: str, primary: bool):
        """Sets the tileset of the header.

        Args:
            label (str): The label of the tileset.
            primary (bool): Whether to set the primary or secondary tileset.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.footer,
            label,
            self.project.config['pymap']['footer'][
                'tileset_primary_path' if primary else 'tileset_secondary_path'
            ],
        )

    def set_tileset_gfx(self, label: str, primary: bool):
        """Sets the gfx of the tileset.

        Args:
            label (str): The label of the gfx.
            primary (bool): Whether to set the primary or secondary tileset.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.tileset_primary if primary else self.tileset_secondary,
            label,
            self.project.config['pymap'][
                'tileset_primary' if primary else 'tileset_secondary'
            ]['gfx_path'],
        )

    def set_map_dimensions(self, width: int, height: int):
        """Sets the map dimensions.

        Args:
            width (int): The width of the map.
            height (int): The height of the map.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.footer,
            width,
            self.project.config['pymap']['footer']['map_width_path'],
        )
        set_member_by_path(
            self.footer,
            height,
            self.project.config['pymap']['footer']['map_height_path'],
        )

    def set_border_dimensions(self, width: int, height: int):
        """Sets the border dimensions.

        Args:
            width (int): The width of the border.
            height (int): The height of the border.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.footer,
            width,
            self.project.config['pymap']['footer']['border_width_path'],
        )
        set_member_by_path(
            self.footer,
            height,
            self.project.config['pymap']['footer']['border_height_path'],
        )

    def set_number_of_connections(self, num_connections: int):
        """Sets the number of connections.

        Args:
            num_connections (int): The number of connections.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.header,
            num_connections,
            self.project.config['pymap']['header']['connections'][
                'connections_size_path'
            ],
        )

    def set_number_of_events(self, event_type: PymapEventConfigType, num_events: int):
        """Sets the number of events.

        Args:
            event_type (PymapEventConfigType): The event type to set
            num_events (int): The number of events.
        """
        assert self.project is not None, 'Project is None'
        set_member_by_path(
            self.header,
            num_events,
            event_type['size_path'],
        )
