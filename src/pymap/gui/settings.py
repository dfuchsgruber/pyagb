"""Settings from the pymap gui by the user."""

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal, overload

import appdirs  # type: ignore

from . import resource_tree

config_file = Path(appdirs.user_config_dir(appname='pymap')) / 'settings.json'  # type: ignore


default_settings: dict[str, Any] = {
    'resource_tree.header_listing': resource_tree.HeaderSorting.NAMESPACE,
    'recent.project': '.',
    'recent.header': '.',
    'recent.footer': '.',
    'recent.tileset': '.',
    'recent.palette': '.',
    'recent.gfx': '.',
    'recent.map_image': '.',
    'map_widget.level_opacity': 30,
    'event_widget.show_pictures': True,
    'connections.mirror_offset': True,
    'tileset.zoom': 20,
}


class Settings(MutableMapping[str, Any]):
    """Class to store settings for pymap."""

    def __init__(self):
        """Initializes the settings for pymap."""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if not config_file.exists():
            # Initially create a configuration
            with open(config_file, 'w+') as f:
                json.dump(default_settings, f)
            self._settings = default_settings
        else:
            self._settings = default_settings.copy()
            with open(config_file) as f:
                override_settings = json.load(f)
                for key in override_settings:
                    self._settings[key] = override_settings[key]

    def __delitem__(self, key: str) -> None:
        """Deletes a setting.

        Args:
            key (str): The key to delete.
        """
        del self._settings[key]
        self.save()

    def __iter__(self):
        """Iterates over the settings."""
        return iter(self._settings)

    def __len__(self):
        """Gets the number of settings."""
        return len(self._settings)

    def __contains__(self, key: Any) -> bool:
        """Checks if a setting exists.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the setting exists, False otherwise.
        """
        return key in self._settings

    @overload
    def __getitem__(
        self, key: Literal['resource_tree.header_listing']
    ) -> resource_tree.HeaderSorting:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.project']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.header']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.footer']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.tileset']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.palette']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.gfx']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['recent.map_image']) -> str:
        ...

    @overload
    def __getitem__(self, key: Literal['map_widget.level_opacity']) -> int:
        ...

    @overload
    def __getitem__(self, key: Literal['event_widget.show_pictures']) -> bool:
        ...

    @overload
    def __getitem__(self, key: Literal['connections.mirror_offset']) -> bool:
        ...

    @overload
    def __getitem__(self, key: Literal['tileset.zoom']) -> int:
        ...

    def __getitem__(self, key: Any) -> Any:
        """Gets a setting.

        Args:
            key (Any): The key to get.

        Returns:
            Any: The value of the setting.
        """
        return self._settings[key]

    @overload
    def __setitem__(
        self,
        key: Literal['resource_tree.header_listing'],
        value: resource_tree.HeaderSorting,
    ) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.project'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.header'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.footer'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.tileset'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.palette'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.gfx'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['recent.map_image'], value: str) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['map_widget.level_opacity'], value: int) -> None:
        ...

    @overload
    def __setitem__(
        self, key: Literal['event_widget.show_pictures'], value: bool
    ) -> None:
        ...

    @overload
    def __setitem__(
        self, key: Literal['connections.mirror_offset'], value: bool
    ) -> None:
        ...

    @overload
    def __setitem__(self, key: Literal['tileset.zoom'], value: int) -> None:
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        """Sets a setting.

        Args:
            key (Any): The key to set.
            value (Any): The value to set.
        """
        self._settings[key] = value
        self.save()

    def save(self):
        """Saves the current settings."""
        with open(config_file, 'w+') as f:
            json.dump(self._settings, f)
