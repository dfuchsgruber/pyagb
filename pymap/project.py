#!/usr/bin/python3

import json
from . import constants, configuration
import os
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
            self.path = None
            self.headers = {}
            self.footers = {}
            self.tilesets_primary = {}
            self.tilesets_secondary = {}
            self.gfxs_primary = {}
            self.gfxs_secondary = {}
            self.constants = constants.Constants({})
            self.config = configuration.default_configuration.copy()
        else:
            os.chdir(os.path.dirname(file_path))
            self.from_file(file_path)
            self.path = file_path

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
        self.tilesets_primary = content['tilesets_primary']
        self.tilesets_secondary = content['tilesets_secondary']
        self.gfxs_primary = content['gfxs_primary']
        self.gfxs_secondary = content['gfxs_secondary']

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
            'tilesets_primary' : self.tilesets_primary,
            'tilesets_secondary' : self.tilesets_secondary,
            'gfxs_primary' : self.gfxs_primary,
            'gfxs_secondary' : self.gfxs_secondary,
        }
        with open(file_path, 'w+') as f:
            json.dump(representation, f, indent=self.config['json']['indent'])
        self.path = file_path


    def unused_banks(self):
        """ Returns a list of all unused map banks. 
        
        Returns:
        --------
        unused_banks : list
            A list of strs, sorted, that holds all unused and therefore free map banks.
        """
        unused_banks = list(range(256))
        for bank in self.headers:
            unused_banks.remove(int(bank))
        return list(map(str, unused_banks))

    def unused_map_idx(self, bank):
        """ Returns a list of all unused map indices in a map bank.
        
        Parameters:
        -----------
        bank : str
            The map bank to scan idx in.
        
        Returns:
        --------
        unused_idx : list
            A list of strs, sorted, that holds all unused and therefore free map indices in this bank.
        """
        unused_idx = list(range(256))
        for idx in self.headers[bank]:
            unused_idx.remove(int(idx))
        return list(map(str, unused_idx))

    def available_namespaces(self):
        """ Returns all available namespaces. If there is a constant table associated with namespaces,
        the choices are restricted to the constant table. Otherwise all maps are scanned and their namespaces
        are returned.

        Returns:
        --------
        namespaces : list
            A list of strs, that holds all namespaces.
        constantized : bool
            If the namespaces are restricted to a set of constants.
        """
        namespace_constants = self.config['pymap']['header']['namespace_constants']
        if namespace_constants is None:
            return list(self.constants[namespace_constants]), True
        else:
            # Scan the entire project
            namespaces = set()
            for bank in self.headers:
                for map_idx in self.headers[bank]:
                    namespaces.add(self.headers[bank][map_idx][2])
            return list(namespaces), False

    def unused_footer_idx(self):
        """ Returns a list of all unused footer indexes sorted. 
        
        Returns:
        --------
        unused_idx : set
            A set of ints, sorted, that holds all unused footer idx.
        """
        unused_idx = set(range(1, 0x10000))
        for footer in self.footers:
            unused_idx.remove(self.footers[footer][0])
        return unused_idx