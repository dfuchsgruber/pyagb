"""Resolving and exporting constants for C and assembly code."""

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator, Literal


class ConstantTable(Mapping[str, int]):
    """A mapping for a constants class.

    This class represents a single constant table, where strings are 
    mapped to numerical values.
    """

    def __init__(self, _type: Literal['dict'] | Literal['enum'],
                 base: int | None=None, values: list[str] | dict[str, int] | None=None):
        """Initializes a constant table.

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
        self.base = base or 0
        if values is not None:
            # Provide a dictionary interface also for enum constants
            if self.type == 'enum':
                if not isinstance(values, list):
                    raise RuntimeError(f'Expected a list for values parameter ' \
                                       f'for type "{self.type}"')
                self._values = {
                    constant: idx + self.base
                    for idx, constant in enumerate(values)
                }
            elif self.type == 'dict':
                if not isinstance(values, dict):
                    raise RuntimeError(f'Expected a dict for values parameter ' \
                                       f'for type "{self.type}"')
                self._values = values
            else:
                raise RuntimeError(f'Unknown constants type "{self.type}"')

    def __getitem__(self, key: str) -> int:
        """Retrieves the value of a constant.

        Args:
            key (str): The constant to retrieve.

        Returns:
            int: The value of the constant.
        """
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        """Iterates over the constants.

        Yields:
            Iterator[str]: The constants.
        """
        return iter(self._values)

    def __len__(self) -> int:
        """Returns the number of constants.

        Returns:
            int: The number of constants.
        """
        return len(self._values)

    def inverse(self) -> dict[int, list[str]]:
        """Returns an inverse mapping of the constants.

        Returns:
            dict[int, list[str]]: The inverse mapping.
        """
        inverse: dict[int, list[str]] = defaultdict(list)
        for k, v in self._values.items():
            inverse[v].append(k)
        return inverse

class Constants:
    """A collection of constant tables."""

    def __init__(self, constant_paths: dict[str, Path]):
        """Lazy constants table initialization.

        Parameters:
        -----------
        constant_paths : dict
            Mapping from constants identifier to the path of the
            constants table. The path is split into its components
            to ensure cross-plattform compatibility.
        """
        self.constant_paths = constant_paths
        # Only initialize a constant table on demand
        self.constant_tables: dict[str, ConstantTable | None] = {
            key : None for key in constant_paths
        }

    def __getitem__(self, key : str) -> ConstantTable:
        """Retrieves a constant table.

        Args:
            key (str): The identifier of the constant table.

        Returns:
            ConstantTable: The constant table.
        """
        if key not in self.constant_tables:
            raise RuntimeError(f'Undefined constant table "{key}"')
        if self.constant_tables[key] is None:
            # Initialize the constant table
            try:
                with open(str(self.constant_paths[key])) as f:
                    content = json.load(f)
                base = None
            except Exception as exn:
                print(f'Could not load constants {key}.')
                raise exn
            if content['type'] == 'enum':
                if 'base' in content:
                    base = content['base']
                else:
                    base = 0
            self.constant_tables[key] = ConstantTable(_type=content['type'], base=base,
                                                      values=content['values'])
        constants_table = self.constant_tables[key]
        if constants_table is None:
            raise RuntimeError(f'Could not load constants "{key}"')
        return constants_table

    def __contains__(self, key: str) -> bool:
        """Checks if a constant table is defined."""
        return key in self.constant_tables
