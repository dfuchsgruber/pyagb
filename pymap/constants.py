#!/usr/bin/python3

""" This module is responsible for resolving constants and exporting them
as C or assembly macros."""

import json
import collections

class ConstantTable(collections.Mapping):
    """ This class represents a single constant table,
     where strings are mapped to numerical values."""

    def __init__(self, _type='dict', base=None, values=None):
        """ Initializes a constant table.
        Parameters:
        -----------
        _type : string in 'dict', 'enum'
            Either the table is a dictionary of string -> int
            or a list of strings, where the mapping string ->
            int is generated iteratively.
        base : int or None
            If the type is 'enum' the base is the integer the
            first element of the constants is assigned to.
        values : dict, enum or None
            The actual values of the constant table.
        """
        self.type = _type
        self.base = base
        if values is not None:
            # Provide a dictionary interface also for enum constants
            if self.type == 'enum':
                if not isinstance(values, list):
                    raise RuntimeError(f'Expected a list for values parameter for type "{self.type}"')
                self.values = dict((constant, idx + self.base) for idx, constant in enumerate(values))
            elif self.type == 'dict':
                if not isinstance(values, dict):
                    raise RuntimeError(f'Expected a dict for values parameter for type "{self.type}"')
                self.values = values
            else:
                raise RuntimeError(f'Unknown constants type "{self.type}"')
        
    def __getitem__(self, key):
        return self.values[key]
    
    def __iter__(self):
        return iter(self.values)
    
    def __len__(self):
        return len(self.values)
        
class Constants:

    def __init__(self, constant_paths):
        """ Initializes a constant instance that can resolve
        mappings from string -> integer serving as constants
        for the pymap project.
        
        Parameters:
        -----------
        constant_paths : dict (string -> path.Path)
            Mapping from constants identifier to the path of the
            constants table. The path is split into its components
            to ensure cross-plattform compatibility.
        """
        self.constant_paths = constant_paths
        # Only initialize a constant table on demand
        self.constant_tables = dict((key, None) for key in constant_paths)

    def __getitem__(self, key):
        if key not in self.constant_tables:
            raise RuntimeError(f'Undefined constant table "{key}"')
        if self.constant_tables[key] is None:
            # Initialize the constant table
            with open(str(self.constant_paths[key])) as f:
                content = json.load(f)
            base = None
            if content['type'] == 'enum':
                if 'base' in content:
                    base = content['base']
                else:
                    base = 0
            self.constant_tables[key] = ConstantTable(_type=content['type'], base=base, values=content['values'])
        return self.constant_tables[key]

    def __contains__(self, key):
        """ Checks if a constant table is defined. """
        return key in self.constant_tables
