#!/usr/bin/python3

import json
from . import constants, configuration
import pymap.model.model
from pathlib import Path
from agb import types
import agb.string.agbstring

class Project:
    """ Represents the central project structure and handles maps, tilesets, gfx... """

    def __init__(self, file_path):
        """ 
        Initializes the project.
        
        Parameters:
        -----------
        file_path : string or None
            The project file path or None (empty project).
        """
        if file_path is None:
            # Initialize empty project
            self.headers = {}
            self.footers = {}
            self.tilesets = {}
            self.gfxs = {}
            self.constants = constants.Constants({})
            self.config = configuration.default_configuration.copy()
        else:
            self.from_file(file_path)

        # Initialize models
        self.model = pymap.model.model.get_model(self.config['model'])

        # Initiaize the string decoder / encoder
        charmap = self.config['string']['charmap']
        if charmap is not None:
            self.coder = agb.string.agbstring.Agbstring(charmap, tail=self.config['string']['tail'])
        else:
            self.coder = None

    def from_file(self, file_path):
        """ Initializes the project from a json file. Should not
        be called manually but only by the constructor of the
        Project class.
        
        Parameters:
        -----------
        file_path : str
            The json file that contains the project information.
        """
        with open(file_path) as f:
            content = json.load(f)

        self.headers = content['headers']
        self.footers = content['footers']
        self.tilesets = content['tilesets']
        self.gfxs = content['gfxs']

        # Initialize the constants
        with open(file_path + '.constants') as f:
            content = json.load(f)
        paths = {key : Path(content[key]) for key in content}
        self.constants = constants.Constants(paths)

        # Initialize the configuration
        self.config = configuration.get_configuration(file_path + '.config')

        
    def save(self, file_path):
        """
        Saves the project to a path.

        Parameters:
        -----------
        file_path : string
            The project file path to save at.
        """
        representation = {
            'headers' : self.headers,
            'footers' : self.footers,
            'tilesets' : self.tilesets,
            'gfxs' : self.gfxs
        }
        with open(file_path, 'w+') as f:
            json.dump(representation, f, indent=self.config['json']['indent'])