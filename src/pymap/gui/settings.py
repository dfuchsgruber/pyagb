"""Settings from the pymap gui by the user."""

import json
from pathlib import Path
from typing import Any

import appdirs  # type: ignore

from . import resource_tree

config_file = Path(appdirs.user_config_dir(appname='pymap')) / 'settings.json' # type: ignore

default_settings = {
    'resource_tree.header_listing' : resource_tree.SORT_BY_NAMESPACE,
    'recent.project' : '.',
    'recent.header' : '.',
    'recent.footer' : '.',
    'recent.tileset' : '.',
    'recent.palette' : '.',
    'recent.gfx' : '.',
    'recent.map_image' : '.',
    'map_widget.level_opacity' : 30,
    'event_widget.show_pictures' : True,
    'connections.mirror_offset' : True,
    'tileset.zoom' : 20,
}

class Settings(dict[str, Any]):
    """Class to store settings for pymap."""

    def __init__(self):
        """Initializes the settings for pymap."""
        super().__init__(**self._load_settings())

    def __setitem__(self, key: str, value: Any):
        """Sets a setting for pymap."""
        super().__setitem__(key, value)
        self._save_settings()

    def __delitem__(self, key: str):
        """Deletes a setting for pymap."""
        del self[key]
        self._save_settings()


    def _load_settings(self) -> dict[str, Any]:
        """Loads the settings for pymap from the last session.

        If this is the first session, default settings are generated and saved.

        Returns:
        -------
        settings : dict[str, any]
        The settings.
        """
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if not config_file.exists():
            # Initially create a configuration
            with open(config_file, 'w+') as f:
                json.dump(default_settings, f)
            return default_settings
        else:
            settings = default_settings.copy()
            with open(config_file) as f:
                override_settings = json.load(f)
                for key in override_settings:
                    settings[key] = override_settings[key]
            return settings


    def _save_settings(self):
        """Saves current settings for pymap.

        Parameters:
        -----------
        settings : dict
            The current pymap settings.
        """
        with open(config_file, 'w+') as f:
            json.dump(self, f)
