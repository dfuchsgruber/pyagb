# Store user settings
import appdirs
import os
import json
import resource_tree
import collections

dir = appdirs.user_config_dir(appname='pymap')
config_file = os.path.join(dir, 'settings.json')

default_settings = {
    'resource_tree.header_listing' : resource_tree.SORT_BY_NAMESPACE,
    'recent.project' : '.',
    'recent.header' : '.',
    'recent.footer' : '.',
    'recent.tileset' : '.',
    'map_widget.level_opacity' : 30,
    'event_widget.show_pictures' : True,
}

class Settings(collections.MutableMapping):
    """ Class to store settings for pymap. """

    def __init__(self):
        self._settings = self._load_settings()

    def __getitem__(self, key):
        return self._settings[key]

    def __setitem__(self, key, value):
        self._settings[key] = value
        self._save_settings()

    def __delitem__(self, key):
        del self._settings[key]
        self._save_settings()

    def __iter__(self):
        return iter(self._settings)

    def __len__(self):
        return len(self._settings)


    def _load_settings(self):
        """ Loads the settings for pymap from the last session. If
        this is the first session, default settings are generated and saved.
        
        Returns:
        -------
        settings : dict
            The settings. """
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        if not os.path.exists(config_file):
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
        """ Saves current settings for pymap. 
        
        Parameters:
        -----------
        settings : dict
            The current pymap settings.
        """
        with open(config_file, 'w+') as f:
            json.dump(self._settings, f)