"""Project structure and handling of maps, tilesets, gfx..."""

from __future__ import annotations

import contextlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Sequence, TypedDict

import agb.string.agbstring
from agb import image
from pymap.gui.properties.utils import get_member_by_path, set_member_by_path
from pymap.gui.smart_shape.smart_shape import SerializedSmartShape
from pymap.gui.smart_shape.template.default_templates import (
    get_default_smart_shape_templates,
)

if TYPE_CHECKING:
    from agb.model.type import Model, ModelValue

    class ModelDataType(TypedDict):
        """Type of the model data."""

        data: ModelValue
        label: str
        type: str


from pymap.configuration import ConfigType

from . import configuration, constants


class HeaderType(NamedTuple):
    """Representation of a map header."""

    label: str | None
    path: str | None
    namespace: str | None


HeadersType = dict[str, dict[str, HeaderType]]


class FooterType(NamedTuple):
    """Representation of a map footer."""

    idx: int
    path: str


FootersType = dict[str, FooterType]
TilesetsType = dict[str, str]
GfxsType = dict[str, str]


class Project:
    """Represents the central project structure and handles maps, tilesets, gfx..."""

    def __init__(self, file_path: str | None | Path):
        """Initializes the project.

        Parameters:
        -----------
        file_path : string or None
            The project file path or None (empty project).
        """
        if file_path is None:
            # Initialize empty project
            self.path = None
            self.headers: HeadersType = {}
            self.footers: FootersType = {}
            self.tilesets_primary: TilesetsType = {}
            self.tilesets_secondary: TilesetsType = {}
            self.gfxs_primary: GfxsType = {}
            self.gfxs_secondary: GfxsType = {}
            self.constants = constants.Constants({})
            self.config: ConfigType = configuration.default_configuration.copy()
            self.smart_shape_templates = get_default_smart_shape_templates()
        else:
            self.from_file(file_path)
            self.path = file_path

        # Initialize models
        import pymap.model.model

        self.model: Model = pymap.model.model.get_model(self.config['model'])

        # Initiaize the string decoder / encoder
        charmap = self.config['string']['charmap']
        if charmap is not None:
            self.coder = agb.string.agbstring.Agbstring(
                charmap, tail=self.config['string']['tail']
            )
        else:
            self.coder = None

    # Getter for self.path
    @property
    def path(self) -> str | Path | None:
        """Returns the file path of the project."""
        return self._path

    @path.setter
    def path(self, value: str | Path | None):
        """Sets the file path of the project."""
        self._path = value
        if value is not None:
            # We cache the project directory as resolution can be costly
            self._project_dir = Path(value).parent.resolve()

    @contextlib.contextmanager
    def project_dir(self):
        """Changes the working directory to the project directory."""
        assert self.path is not None, 'Project path is not initialized'
        assert self._project_dir is not None, 'Project directory is not initialized'
        yield self._project_dir
        # with working_dir(self._project_dir) as path:
        #     yield path

    def from_file(self, file_path: str | Path):
        """Initializes the project from a json file.

        Should not be called manually but only by the constructor of the Project class.

        Parameters:
        -----------
        file_path : str
            The json file that contains the project information.
        """
        with open(Path(file_path)) as f:
            content = json.load(f)

        self.headers = content['headers']
        for bank in self.headers:
            self.headers[bank] = {
                map_idx: HeaderType(*self.headers[bank][map_idx])
                for map_idx in self.headers[bank]
            }

        self.footers = content['footers']
        for label in self.footers:
            self.footers[label] = FooterType(*self.footers[label])
        self.tilesets_primary = content['tilesets_primary']
        self.tilesets_secondary = content['tilesets_secondary']
        self.gfxs_primary = content['gfxs_primary']
        self.gfxs_secondary = content['gfxs_secondary']

        # Initialize the constants
        with open(str(Path(file_path)) + '.constants') as f:
            content = json.load(f)
        paths = {key: Path(content[key]) for key in content}
        self.constants: constants.Constants = constants.Constants(paths)

        # Initialize the configuration
        self.config = configuration.get_configuration(str(file_path) + '.config')
        self.smart_shape_templates = get_default_smart_shape_templates()

    def autosave(self):
        """Saves the project if it is stated in the configuration."""
        if self.config['pymap']['project']['autosave']:
            assert self.path is not None, 'Project path is not initialized'
            self.save(self.path)

    def save(self, file_path: str | Path):
        """Saves the project to a path.

        Parameters:
        -----------
        file_path : string
            The project file path to save at.
        """
        representation: dict[str, Any] = {
            'headers': self.headers,
            'footers': self.footers,
            'tilesets_primary': self.tilesets_primary,
            'tilesets_secondary': self.tilesets_secondary,
            'gfxs_primary': self.gfxs_primary,
            'gfxs_secondary': self.gfxs_secondary,
        }
        with open(Path(file_path), 'w+') as f:
            json.dump(representation, f, indent=self.config['json']['indent'])
        self.path = file_path

    def load_header(
        self, bank: int | str, map_idx: int | str, unpack_connections: bool = False
    ) -> tuple[ModelValue, str | None, str | None]:
        """Opens a header by its location in the table and verifies label and type.

        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is
            converted into its "canonical" form.
        unpack_connections : bool
            If the connections should be unpacked, i.e. if the blocks of
            connections should also be loaded.

        Returns:
        --------
        header : dict or None
            The header instance
        label : str or None
            The label of the header
        namespace : str or None
            The namespace of the header
        """
        from pymap.gui.properties import get_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
            if bank in self.headers and map_idx in self.headers[bank]:
                label, path, namespace = self.headers[bank][map_idx]
                assert path is not None, (
                    f'Path to header [{bank}, {map_idx.zfill(2)}] is not initialized'
                )
                with open(Path(path), encoding=self.config['json']['encoding']) as f:
                    header: ModelDataType = json.load(f)
                assert header['label'] == label, (
                    'Header label mismatches the label stored in the project file'
                )
                assert header['type'] == self.config['pymap']['header']['datatype'], (
                    'Header datatype mismatches the configuration'
                )
                assert namespace is not None, (
                    f'Namespace of header [{bank}, {map_idx.zfill(2)}] is not defined'
                )
                namespace_in_data = get_member_by_path(
                    header['data'], self.config['pymap']['header']['namespace_path']
                )
                assert isinstance(namespace_in_data, str), (
                    f'Namespace of header [{bank}, {map_idx.zfill(2)}] is not a string'
                )
                assert _canonical_form(namespace) == _canonical_form(
                    namespace_in_data
                ), (
                    f'Header {bank}.{map_idx} namespace mismatches '
                    'the namespaces stored in the project file'
                )
                data = header['data']
                if unpack_connections:
                    from pymap.gui.blocks import (
                        unpack_connections as _unpack_connections,
                    )

                    connections = get_member_by_path(
                        data,
                        self.config['pymap']['header']['connections'][
                            'connections_path'
                        ],
                    )
                    unpacked_connections = _unpack_connections(connections, self)
                    set_member_by_path(
                        data,
                        unpacked_connections,
                        self.config['pymap']['header']['connections'][
                            'connections_path'
                        ],
                    )

                return data, label, namespace
            else:
                return None, None, None

    def refactor_header(
        self, bank: str | int, map_idx: str | int, label: str, namespace: str
    ):
        """Changes the label and namespace of a map header.

        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is
            converted into its "canonical" form.
        label : str
            The new label.
        namespace : str
            The new namespace.
        """
        from pymap.gui.properties import set_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            if bank in self.headers and map_idx in self.headers[bank]:
                path = self.headers[bank][map_idx][1]
                assert path is not None, (
                    f'Path to header [{bank}, {map_idx.zfill(2)}] is not set'
                )
                header, _, _ = self.load_header(bank, map_idx)
                set_member_by_path(
                    header, namespace, self.config['pymap']['header']['namespace_path']
                )
                self.headers[bank][map_idx] = HeaderType(label, path, namespace)
                self.save_header(header, bank, map_idx)
                self.autosave()
            else:
                raise RuntimeError(f'Header [{bank}, {map_idx}] not existent.')

    def import_header(
        self,
        bank: int | str,
        map_idx: int | str,
        label: str,
        path: str,
        namespace: str,
        footer: str,
    ):
        """Imports a header structure into the project.

        This will change label and namespace and footer of the json file.

        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is
            converted into its "canonical" form.
        label : str
            The new label.
        path : str
            Path to the map header structure.
        namespace : str
            The new namespace.
        footer : str
            The footer of the new map.
        """
        from pymap.gui.properties import set_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
            if bank in self.headers:
                if map_idx not in self.headers[bank]:
                    with open(
                        Path(path), encoding=self.config['json']['encoding']
                    ) as f:
                        header = json.load(f)
                    assert (
                        header['type'] == self.config['pymap']['header']['datatype']
                    ), 'Header datatype mismatches the configuration'
                    self.headers[bank][map_idx] = HeaderType(
                        label, os.path.relpath(path), namespace
                    )
                    if footer not in self.footers:
                        raise RuntimeError(f'Footer {footer} is not existent.')
                    set_member_by_path(
                        header['data'],
                        namespace,
                        self.config['pymap']['header']['namespace_path'],
                    )
                    set_member_by_path(
                        header['data'],
                        footer,
                        self.config['pymap']['header']['footer_path'],
                    )
                    set_member_by_path(
                        header['data'],
                        int(self.footers[footer][0]),
                        self.config['pymap']['header']['footer_idx_path'],
                    )
                    self.save_header(header['data'], bank, map_idx)
                    self.autosave()
                else:
                    raise RuntimeError(
                        f'Header [{bank}, {map_idx.zfill(2)}] already exists.'
                    )
            else:
                raise RuntimeError(f'Bank {bank} not existent. ')

    def remove_bank(self, bank: int | str):
        """Removes an entire mapbank from the project.

        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is
            converted into its "canonical" form.
        """
        bank = _canonical_form(bank)
        if bank in self.headers:
            del self.headers[bank]
            self.autosave()
        else:
            raise RuntimeError(f'Bank {bank} not existent.')

    def remove_header(self, bank: str | int, map_idx: str | int):
        """Removes a map header from the project.

        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string,
            it is converted into its "canonical" form.
        """
        bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
        if bank in self.headers and map_idx in self.headers[bank]:
            del self.headers[bank][map_idx]
            self.autosave()
        else:
            raise RuntimeError(f'Header [{bank}, {map_idx.zfill(2)}] not existent.')

    def save_header(
        self,
        header: ModelValue,
        bank: str | int,
        map_idx: str | int,
        pack_connections: bool = False,
    ):
        """Saves a header.

        Parameters:
        -----------
        header : dict
            The header to save.
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is
            converted into its "canonical" form.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
            if bank in self.headers and map_idx in self.headers[bank]:
                label, path, _ = self.headers[bank][map_idx]
                assert path is not None, (
                    f'Path to header [{bank}, {map_idx.zfill(2)}] is not initialized'
                )
                if pack_connections:
                    from pymap.gui.blocks import (
                        pack_connections as _pack_connections,
                    )

                    header = deepcopy(header)
                    connections = get_member_by_path(
                        header,
                        self.config['pymap']['header']['connections'][
                            'connections_path'
                        ],
                    )
                    assert isinstance(connections, list), (
                        'Connections are not a list, cannot pack them'
                    )
                    set_member_by_path(
                        header,
                        _pack_connections(connections, self),
                        self.config['pymap']['header']['connections'][
                            'connections_path'
                        ],
                    )

                with open(
                    Path(path), 'w+', encoding=self.config['json']['encoding']
                ) as f:
                    json.dump(
                        {
                            'data': header,
                            'label': label,
                            'type': self.config['pymap']['header']['datatype'],
                        },
                        f,
                        indent=self.config['json']['indent'],
                    )
            else:
                raise RuntimeError(f'No header located at [{bank}, {map_idx}]')

    def new_header(
        self, label: str, path: str, namespace: str, bank: int | str, map_idx: int | str
    ) -> ModelValue:
        """Creates a new header, assigns the namespace and saves it into a file.

        Parameters:
        -----------
        label : str
            The label of the header.
        path : str
            Path to the header structure file.
        namespace : str
            The namespace of the header.
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into
            its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is
            converted into its "canonical" form.

        Returns:
        --------
        header : ModelValue
            The new header.
        """
        from pymap.gui.properties import set_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
            if bank in self.headers:
                if map_idx in self.headers[bank]:
                    raise RuntimeError(
                        f'Index {map_idx} already present in bank {bank}'
                    )
                else:
                    self.headers[bank][map_idx] = HeaderType(
                        label, os.path.relpath(path), namespace
                    )
                    datatype = self.config['pymap']['header']['datatype']
                    header = self.model[datatype](self, [], [])
                    # Assign the proper namespace
                    set_member_by_path(
                        header,
                        namespace,
                        self.config['pymap']['header']['namespace_path'],
                    )
                    # Save the header
                    self.save_header(header, bank, map_idx)
                    self.autosave()
                    return header
            else:
                raise RuntimeError(f'Bank {bank} not existent')

    def refactor_footer(self, label_old: str, label_new: str):
        """Changes the label of a footer.

        Applies changes to all headers refering to this footer.

        Parameters:
        -----------
        label_old : str
            The label of the footer to change.
        label_new : str
            The new label of the footer.
        """
        from pymap.gui.properties import get_member_by_path, set_member_by_path

        assert label_old in self.footers, f'Footer {label_old} not existent.'
        assert label_new not in self.footers, (
            f'Footer label {label_new} already existent.'
        )
        for bank in self.headers:
            for map_idx in self.headers[bank]:
                header, _, _ = self.load_header(bank, map_idx)
                if (
                    get_member_by_path(
                        header, self.config['pymap']['header']['footer_path']
                    )
                    == label_old
                ):
                    set_member_by_path(
                        header, label_new, self.config['pymap']['header']['footer_path']
                    )
                    self.save_header(header, bank, map_idx)
                    print(
                        f'Refactored footer reference in header '
                        f'[{bank}, {map_idx.zfill(2)}]'
                    )
        footer, _, smart_shapes = self.load_footer(label_old)
        assert smart_shapes is not None, 'Smart shapes not found.'
        self.footers[label_new] = self.footers.pop(label_old)
        self.save_footer(footer, label_new, smart_shapes)
        self.autosave()

    def remove_footer(self, label: str):
        """Removes a footer form the project.

        Parameters:
        -----------
        label : str
            The label of the footer to load.
        """
        if label in self.footers:
            del self.footers[label]
            self.autosave()
        else:
            raise RuntimeError(f'Footer {label} non existent.')

    def load_footer(
        self,
        label: str,
        map_blocks_to_ndarray: bool = False,
        border_blocks_to_ndarray: bool = False,
    ) -> (
        tuple[ModelValue, int, dict[str, SerializedSmartShape]]
        | tuple[None, Literal[-1], dict[str, SerializedSmartShape]]
    ):
        """Opens a footer by its label and verifies the label and type of json instance.

        Parameters:
        -----------
        label : str
            The label of the footer to load.
        map_blocks_to_ndarray : bool
            If the map blocks should be converted to a numpy array.
        border_blocks_to_ndarray : bool
            If the border blocks should be converted to a numpy array.

        Returns:
        --------
        footer : dict or None
            The footer instance if present.
        footer_idx : int
            The index of the footer or -1 if no footer is present.
        """
        from pymap.gui.blocks import blocks_to_ndarray
        from pymap.gui.properties import get_member_by_path, set_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            if label in self.footers:
                footer_idx, path = self.footers[label]
                with open(Path(path), encoding=self.config['json']['encoding']) as f:
                    footer = json.load(f)
                assert footer['label'] == label, (
                    'Footer label mismatches the label stored in the project.'
                )
                assert footer['type'] == self.config['pymap']['footer']['datatype'], (
                    'Footer datatype mismatches the configuration'
                )
                smart_shapes: dict[str, SerializedSmartShape] = footer.get(
                    'smart_shapes', {}
                )
                footer = footer['data']
                assert isinstance(footer, dict), 'Footer is not a dictionary'
                if map_blocks_to_ndarray:
                    map_blocks = blocks_to_ndarray(
                        get_member_by_path(
                            footer,  # type: ignore
                            self.config['pymap']['footer']['map_blocks_path'],
                        )
                    )
                    set_member_by_path(
                        footer,  # type: ignore
                        map_blocks,
                        self.config['pymap']['footer']['map_blocks_path'],
                    )
                if border_blocks_to_ndarray:
                    border_blocks = blocks_to_ndarray(
                        get_member_by_path(
                            footer,  # type: ignore
                            self.config['pymap']['footer']['border_path'],
                        )
                    )
                    set_member_by_path(
                        footer,  # type: ignore
                        border_blocks,
                        self.config['pymap']['footer']['border_path'],
                    )
                return footer, footer_idx, smart_shapes  # type: ignore
            else:
                return None, -1, {}

    def save_footer(
        self,
        footer: ModelValue,
        label: str,
        smart_shapes: dict[str, SerializedSmartShape],
        map_blocks_to_list: bool = False,
        border_blocks_to_list: bool = False,
    ):
        """Saves a footer.

        Parameters:
        -----------
        footer : dict
            The footer to save.
        label : str
            The label of the footer.
        """
        assert self.path is not None, 'Project path is not initialized'
        assert isinstance(footer, dict), 'Footer is not a dictionary'
        if map_blocks_to_list:
            from pymap.gui.blocks import ndarray_to_blocks

            footer = deepcopy(footer)

            map_blocks = ndarray_to_blocks(
                get_member_by_path(
                    footer, self.config['pymap']['footer']['map_blocks_path']
                )  # type: ignore
            )
            set_member_by_path(
                footer, map_blocks, self.config['pymap']['footer']['map_blocks_path']
            )
        if border_blocks_to_list:
            from pymap.gui.blocks import ndarray_to_blocks

            footer = deepcopy(footer)

            border_blocks = ndarray_to_blocks(
                get_member_by_path(
                    footer,
                    self.config['pymap']['footer']['border_path'],
                )  # type: ignore
            )
            set_member_by_path(
                footer, border_blocks, self.config['pymap']['footer']['border_path']
            )

        with self.project_dir():
            if label in self.footers:
                _, path = self.footers[label]
                with open(
                    Path(path), 'w+', encoding=self.config['json']['encoding']
                ) as f:
                    json.dump(
                        {
                            'data': footer,
                            'label': label,
                            'type': self.config['pymap']['footer']['datatype'],
                            'smart_shapes': smart_shapes,
                        },
                        f,
                        indent=self.config['json']['indent'],
                    )
            else:
                raise RuntimeError(f'No footer {label}')

    def new_footer(self, label: str, path: str, footer_idx: int) -> ModelValue:
        """Creates a new footer.

        Parameters:
        -----------
        label : str
            The label of the header.
        path : str
            Path to the header structure file.
        namespace : str
            The namespace of the header.
        footer_idx : int or str
            The index in the footer table.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            if label in self.footers:
                raise RuntimeError(f'Footer {label} already present.')
            elif footer_idx not in self.unused_footer_idx():
                raise RuntimeError(f'Footer index {footer_idx} already present.')
            else:
                self.footers[label] = FooterType(footer_idx, os.path.relpath(path))
                datatype = self.config['pymap']['footer']['datatype']
                footer = self.model[datatype](self, [], [])
                # Save the footer
                self.save_footer(footer, label, {})
                self.autosave()
                return footer

    def import_footer(self, label: str, path: str, footer_idx: int):
        """Imports a new footer.

        Parameters:
        -----------
        label : str
            The label of the footer. The json file will be modified s.t.
            the labels match.
        path : str
            Path to the footer file.
        footer_idx : int
            Index of the footer.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            if label in self.footers:
                raise RuntimeError(f'Footer {label} already existent.')
            if footer_idx not in self.unused_footer_idx():
                raise RuntimeError(f'Footer index {footer_idx} already in use.')
            with open(Path(path), encoding=self.config['json']['encoding']) as f:
                footer = json.load(f)
            assert footer['type'] == self.config['pymap']['footer']['datatype'], (
                'Footer datatype mismatches the configuration'
            )
            self.footers[label] = FooterType(footer_idx, os.path.relpath(path))
            self.save_footer(footer['data'], label, footer.get('smart_shapes', []))
            self.autosave()

    def remove_tileset(self, primary: bool, label: str):
        """Removes a tileset from the project.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset.
        label : str
            The label of the tileset.
        """
        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        if label in tilesets:
            del tilesets[label]
            self.autosave()
        else:
            raise RuntimeError(f'Tileset {label} not existent')

    def load_tileset(self, primary: bool, label: str) -> ModelValue:
        """Loads a tileset by its label and verifies label and type.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset.
        label : str
            The label of the tileset.

        Returns:
        --------
        tileset : dict or None
            The tileset structure.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            path = (self.tilesets_primary if primary else self.tilesets_secondary).get(
                label, None
            )
            if path is None:
                return None
            else:
                assert path is not None
                with open(Path(path), encoding=self.config['json']['encoding']) as f:
                    tileset = json.load(f)
                assert tileset['label'] == label, (
                    'Tileset label mismatches the label stored in the project'
                )
                assert (
                    tileset['type']
                    == self.config['pymap'][
                        ('tileset_primary' if primary else 'tileset_secondary')
                    ]['datatype']
                ), 'Tileset datatype mismatches the configuration'
                return tileset['data']

    def save_tileset(self, primary: bool, tileset: ModelValue, label: str):
        """Saves a tileset.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset.
        tileset : dict
            The tileset structure.
        label : str
            The label of the tileset.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            tilesets = self.tilesets_primary if primary else self.tilesets_secondary
            if label in tilesets:
                path = tilesets[label]
                with open(
                    Path(path), 'w+', encoding=self.config['json']['encoding']
                ) as f:
                    json.dump(
                        {
                            'data': tileset,
                            'label': label,
                            'type': self.config['pymap'][
                                'tileset_primary' if primary else 'tileset_secondary'
                            ]['datatype'],
                        },
                        f,
                        indent=self.config['json']['indent'],
                    )
            else:
                raise RuntimeError(f'No tileset {label}')

    def new_tileset(
        self,
        primary: bool,
        label: str,
        path: str,
        gfx_compressed: bool = True,
        tileset: ModelValue | None = None,
    ) -> ModelValue:
        """Creates a new tileset.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset.
        label : str
            The label of the tileset.
        path : str
            Path to the tileset structure.
        gfx_compressed : bool
            If the gfx is expected to be compressed in the ROM.
        tileset : optional, dict
            The new tileset. If not given, an empty default tileset is created from the
            data model.
        """
        from pymap.gui.properties import set_member_by_path

        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            tilesets = self.tilesets_primary if primary else self.tilesets_secondary
            if label in tilesets:
                raise RuntimeError(f'Tileset {label} already present.')
            else:
                tilesets[label] = os.path.relpath(path)
                config = self.config['pymap'][
                    'tileset_primary' if primary else 'tileset_secondary'
                ]
                if tileset is None:
                    datatype = config['datatype']
                    tileset = self.model[datatype](self, [], [])
                    set_member_by_path(
                        tileset, str(int(gfx_compressed)), config['gfx_compressed_path']
                    )
                # Save the tileset
                self.save_tileset(primary, tileset, label)
                self.autosave()
                return tileset

    def refactor_tileset(self, primary: bool, label_old: str, label_new: str):
        """Changes the label of a tileset.

        Applies changes to all footers refering to this tileset.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset or not (secondary tileset).
        label_old : str
            The old label of the tileset to refactor.
        label_new : str
            Its new label.
        """
        from pymap.gui.properties import get_member_by_path, set_member_by_path

        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        assert label_old in tilesets, f'Tileset {label_old} not existent.'
        assert label_new not in tilesets, f'Tileset {label_new} already existent.'
        for label in self.footers:
            footer, footer_idx, smart_shapes = self.load_footer(
                label, border_blocks_to_ndarray=False, map_blocks_to_ndarray=False
            )
            assert footer_idx != -1, f'Footer {label} not existent.'
            assert smart_shapes is not None, 'Smart shapes not initialized.'
            if (
                get_member_by_path(
                    footer,
                    self.config['pymap']['footer'][
                        'tileset_primary_path' if primary else 'tileset_secondary_path'
                    ],
                )
                == label_old
            ):
                set_member_by_path(
                    footer,
                    label_new,
                    self.config['pymap']['footer'][
                        'tileset_primary_path' if primary else 'tileset_secondary_path'
                    ],
                )
                self.save_footer(footer, label, smart_shapes)
                print(f'Refactored tileset reference in footer {label}')
        tileset = self.load_tileset(primary, label_old)
        tilesets[label_new] = tilesets.pop(label_old)
        self.save_tileset(primary, tileset, label_new)
        self.autosave()

    def import_tileset(self, primary: bool, label: str, path: str):
        """Imports a tileset into the project.

        Parameters:
        -----------
        primary : bool
            If the tileset will be a primary tileset.
        label : str
            The label of the tileset.
        path : str
            The path to the tileset.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            tilesets = self.tilesets_primary if primary else self.tilesets_secondary
            if label in tilesets:
                raise RuntimeError(f'Tileset {label} already existent.')
            with open(Path(path), encoding=self.config['json']['encoding']) as f:
                tileset = json.load(f)
            assert (
                tileset['type']
                == self.config['pymap'][
                    'tileset_primary' if primary else 'tileset_secondary'
                ]['datatype']
            ), 'Tileset datatype mismatches the configuration'
            tilesets[label] = os.path.relpath(path)
            self.save_tileset(primary, tileset['data'], label)
            self.autosave()

    def load_gfx(self, primary: bool, label: str) -> image.Image:
        """Loads a gfx and instanciates an agb image.

        Parameters:
        -----------
        primary : bool
            If the image is a gfx for a primary or secondary tileset.
        label : str
            The label the gfx is associated with.

        Returns:
        --------
        image : agb.image.Image
            The agb image.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            gfx = self.gfxs_primary if primary else self.gfxs_secondary
            if label not in gfx:
                raise RuntimeError(f'No gfx associated with label {gfx}')
            else:
                path = gfx[label]
                img, _ = image.from_file(path)
                return img

    def save_gfx(
        self, primary: bool, image: image.Image, palette: Sequence[int], label: str
    ):
        """Saves a gfx with a certain palette.

        Parameters:
        -----------
        primary : bool
            If the image is a gfx for a primary or secondary tileset.
        image : agb.image
            The agb image of the gfx.
        palette : agb.palette
            The agb palette to save the gfx in.
        label : str
            The label the gfx is associated with.
        """
        assert self.path is not None, 'Project path is not initialized'
        with self.project_dir():
            gfx = self.gfxs_primary if primary else self.gfxs_secondary
            if label not in gfx:
                raise RuntimeError(f'No gfx associated with label {gfx}')
            else:
                path = gfx[label]
                image.save(path, palette)

    def refactor_gfx(self, primary: bool, label_old: str, label_new: str):
        """Changes the label of a gfx.

        Applies changes to all tilesets refering to this gfx.

        Parameters:
        -----------
        primary : bool
            If the tileset is a primary gfx or not (secondary gfx).
        label_old : str
            The old label of the gfx to refactor.
        label_new : str
            Its new label.
        """
        from pymap.gui.properties import get_member_by_path, set_member_by_path

        gfxs = self.gfxs_primary if primary else self.gfxs_secondary
        assert label_old in gfxs, f'Gfx {label_old} not existent.'
        assert label_new not in gfxs, f'Gfx {label_new} already existent.'
        for label in self.tilesets_primary if primary else self.tilesets_secondary:
            tileset = self.load_tileset(primary, label)
            if (
                get_member_by_path(
                    tileset,
                    self.config['pymap'][
                        'tileset_primary' if primary else 'tileset_secondary'
                    ]['gfx_path'],
                )
                == label_old
            ):
                set_member_by_path(
                    tileset,
                    label_new,
                    self.config['pymap'][
                        'tileset_primary' if primary else 'tileset_secondary'
                    ]['gfx_path'],
                )
                self.save_tileset(primary, tileset, label)
                print(f'Refactored gfx reference in tileset {label}')
        gfxs[label_new] = gfxs.pop(label_old)
        self.autosave()

    def remove_gfx(self, primary: bool, label: str):
        """Removes a gfx from the project.

        Parameters:
        -----------
        primary : bool
            If the gfx is a primary gfx.
        label : str
            The label of the gfx.
        """
        gfxs = self.gfxs_primary if primary else self.gfxs_secondary
        if label in gfxs:
            del gfxs[label]
            self.autosave()
        else:
            raise RuntimeError(f'Gfx {label} not existent')

    def import_gfx(self, primary: bool, label: str, path: str):
        """Imports a gfx into the project.

        Assertions on the bitdepth and image size are performed.

        Parameters:
        -----------
        primary : bool
            If the gfx is a primary gfx.
        label : str
            The label of the gfx.
        path : str
            The path to the gfx.
        """
        gfxs = self.gfxs_primary if primary else self.gfxs_secondary
        if label in gfxs:
            raise RuntimeError(f'Gfx {label} already exists.')
        else:
            # Load gfx and assert size bounds
            img, _ = image.from_file(path)
            assert img.depth == 4
            assert img.width % 8 == 0
            assert img.height % 8 == 0
            if primary:
                assert img.width * img.height == 320 * 128
            else:
                assert img.width * img.height == 192 * 128
            gfxs[label] = os.path.relpath(path)
            self.autosave()

    def unused_banks(self):
        """Returns a list of all unused map banks.

        Returns:
        --------
        unused_banks : list
            A list of strs, sorted, that holds all unused and therefore free map banks.
        """
        unused_banks = list(range(256))
        for bank in self.headers:
            unused_banks.remove(int(bank))
        return list(map(str, unused_banks))

    def unused_map_idx(self, bank: str | int):
        """Returns a list of all unused map indices in a map bank.

        Parameters:
        -----------
        bank : str
            The map bank to scan idx in.

        Returns:
        --------
        unused_idx : list
            A list of strs, sorted, that holds all unused and therefore free
            map indices in this bank.
        """
        unused_idx = list(range(256))
        for idx in self.headers[_canonical_form(bank)]:
            unused_idx.remove(int(idx))
        return list(map(str, unused_idx))

    def available_namespaces(self) -> tuple[list[str], bool]:
        """Returns all available namespaces.

        If there is a constant table associated with namespaces,
        the choices are restricted to the constant table. Otherwise all
        maps are scanned and their namespaces are returned.

        Returns:
        --------
        namespaces : list
            A list of strs, that holds all namespaces.
        constantized : bool
            If the namespaces are restricted to a set of constants.
        """
        namespace_constants = self.config['pymap']['header']['namespace_constants']
        if namespace_constants is not None:
            return list(self.constants[namespace_constants]), True
        else:
            # Scan the entire project
            namespaces: set[str] = set()
            for bank in self.headers:
                for map_idx in self.headers[bank]:
                    bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
                    namespace = self.headers[bank][map_idx][2]
                    if namespace:
                        namespaces.add(namespace)
            return list(namespaces), False

    def unused_footer_idx(self) -> set[int]:
        """Returns a list of all unused footer indexes sorted.

        Returns:
        --------
        unused_idx : set
            A set of ints, sorted, that holds all unused footer idx.
        """
        unused_idx = set(range(1, 0x10000))
        for footer in self.footers:
            unused_idx.remove(self.footers[footer][0])
        return unused_idx


def _canonical_form(x: str | int) -> str:
    try:
        return str(int(str(x), 0))
    except ValueError:
        return str(x)


@contextlib.contextmanager
def working_dir(file_path: Path | os.PathLike[str]):
    """Changes the working directory to the directory of the file path.

    Parameters:
    -----------
    file_path : str or Path
        The file path to change the working directory to.
    """
    saved_dir = Path(os.getcwd())
    path = Path(file_path).resolve()
    assert path.is_dir, f'{path} is not a directory'
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(saved_dir)
