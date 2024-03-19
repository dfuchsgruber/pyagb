"""Settings from the pymap gui by the user."""

import json
from pathlib import Path
from typing import TypedDict

import appdirs  # type: ignore

from . import resource_tree

config_file = Path(appdirs.user_config_dir(appname='pymap')) / 'settings.json'  # type: ignore


class SettingsDict(TypedDict):
    """Type for the settings dictionary."""

    resource_tree_header_listing: resource_tree.HeaderSorting
    recent_project: str
    recent_header: str
    recent_footer: str
    recent_tileset: str
    recent_palette: str
    recent_gfx: str
    recent_map_image: str
    map_widget_level_opacity: int
    event_widget_show_pictures: bool
    connections_mirror_offset: bool
    tileset_zoom: int


default_settings: SettingsDict = SettingsDict(
    resource_tree_header_listing=resource_tree.HeaderSorting.NAMESPACE,
    recent_project='.',
    recent_header='.',
    recent_footer='.',
    recent_tileset='.',
    recent_palette='.',
    recent_gfx='.',
    recent_map_image='.',
    map_widget_level_opacity=30,
    event_widget_show_pictures=True,
    connections_mirror_offset=True,
    tileset_zoom=20,
)


class Settings:
    """Class to store settings for pymap."""

    def __init__(self):
        """Initializes the settings for pymap."""
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if not config_file.exists():
            # Initially create a configuration
            with open(config_file, 'w+') as f:
                json.dump(default_settings, f)
            self.settings = default_settings
        else:
            self.settings = default_settings.copy()
            with open(config_file) as f:
                override_settings = json.load(f)
                for key in override_settings:
                    self.settings[key] = override_settings[key]

    def save(self):
        """Saves the current settings."""
        with open(config_file, 'w+') as f:
            json.dump(self.settings, f)
