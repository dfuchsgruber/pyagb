#!/usr/bin/python3

import json
from . import constants, configuration
from pathlib import Path
from agb import types

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
            self.banks = {}
            self.footers = {}
            self.tilesets = {}
            self.images = {}
            self.constants = constants.Constants({})
            self.config = configuration.default_configuration.copy()
        else:
            self.from_file(file_path)

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

        self.banks = content['banks']
        self.footers = content['footers']
        self.tilesets = content['tilesets']
        self.images = content['images']

        # Initialize the constants
        with open(file_path + '.constants') as f:
            content = json.load(f)
        paths = {key : Path(content[key]) for key in content}
        self.constants = constants.Constants(paths)

        # Initialize the configuration
        self.conifg = configuration.get_configuration(file_path + '.config')

        
    def save(self, file_path):
        """
        Saves the project to a path.

        Parameters:
        -----------
        file_path : string
            The project file path to save at.
        """
        representation = {
            'banks' : self.banks,
            'footers' : self.footers,
            'tilesets' : self.tilesets,
            'images' : self.images
        }
        with open(file_path, 'w+') as f:
            json.dump(representation, f, indent='\t')