#!/usr/bin/python3

import json
from . import constants, configuration
from pymap.gui.properties import set_member_by_path, get_member_by_path
import os
import pymap.model.model
from pathlib import Path
from agb import types, image
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
            os.chdir(os.path.abspath(os.path.dirname(file_path)))
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
        with open(Path(file_path)) as f:
            content = json.load(f)

        self.headers = content['headers']
        self.footers = content['footers']
        self.tilesets_primary = content['tilesets_primary']
        self.tilesets_secondary = content['tilesets_secondary']
        self.gfxs_primary = content['gfxs_primary']
        self.gfxs_secondary = content['gfxs_secondary']

        # Initialize the constants
        with open(str(Path(file_path)) + '.constants') as f:
            content = json.load(f)
        paths = {key : Path(content[key]) for key in content}
        self.constants = constants.Constants(paths)

        # Initialize the configuration
        self.config = configuration.get_configuration(file_path + '.config')

    def autosave(self):
        """ Saves the project if it is stated in the configuration. """
        if self.config['pymap']['project']['autosave']:
            self.save(self.path)

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
        with open(Path(file_path), 'w+') as f:
            json.dump(representation, f, indent=self.config['json']['indent'])
        self.path = file_path

    def load_header(self, bank, map_idx):
        """ Opens a header by its location in the table and verifies label and type of the json instance.
        
        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.

        Returns:
        --------
        header : dict or None
            The header instance
        label : str or None
            The label of the header
        namespace : str or None
            The namespace of the header
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
        if bank in self.headers and map_idx in self.headers[bank]:
            label, path, namespace = self.headers[bank][map_idx]
            with open(Path(path), encoding=self.config['json']['encoding']) as f:
                header = json.load(f)
            assert(header['label'] == label), f'Header label mismatches the label stored in the project file'
            assert(header['type'] == self.config['pymap']['header']['datatype']), f'Header datatype mismatches the configuration'
            assert(_canonical_form(namespace) == _canonical_form(get_member_by_path(header['data'], self.config['pymap']['header']['namespace_path']))), f'Header {bank}.{map_idx} namespace mismatches the namespaces stored in the project file'
            return header['data'], label, namespace
        else:
            return None, None, None

    def refactor_header(self, bank, map_idx, label, namespace):
        """ Changes the label and namespace of a map header. 
        
        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.
        label : str
            The new label.
        namespace : str
            The new namespace.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if bank in self.headers and map_idx in self.headers[bank]:
            path = self.headers[bank][map_idx][1]
            header, _, _ = self.load_header(bank, map_idx)
            set_member_by_path(header, namespace, self.config['pymap']['header']['namespace_path'])
            self.headers[bank][map_idx] = label, path, namespace
            self.save_header(header, bank, map_idx)
            self.autosave()
        else:
            raise RuntimeError(f'Header [{bank}, {map_idx.zfill(2)}] not existent.')

    def import_header(self, bank, map_idx, label, path, namespace, footer):
        """ Imports a header structure into the project. This will change label and namespace and footer of the json file.
        
        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.
        label : str
            The new label.
        path : str
            Path to the map header structure. 
        namespace : str
            The new namespace.
        footer : str
            The footer of the new map.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if bank in self.headers:
            if not map_idx in self.headers[bank]:
                with open(Path(path), encoding=self.config['json']['encoding']) as f:
                    header = json.load(f)
                assert(header['type'] == self.config['pymap']['header']['datatype']), 'Header datatype mismatches the configuration'
                self.headers[bank][map_idx] = [label, os.path.relpath(path), namespace]
                if footer not in self.footers:
                    raise RuntimeError(f'Footer {footer} is not existent.')
                set_member_by_path(header['data'], namespace, self.config['pymap']['header']['namespace_path'])
                set_member_by_path(header['data'], footer, self.config['pymap']['header']['footer_path'])
                set_member_by_path(header['data'], int(self.footers[footer][0]), self.config['pymap']['header']['footer_idx_path'])
                self.save_header(header['data'], bank, map_idx)
                self.autosave()
            else:
                raise RuntimeError(f'Header [{bank}, {map_idx.zfill(2)}] already exists.')
        else:
            raise RuntimeError(f'Bank {bank} not existent. ')

    def remove_bank(self, bank):
        """ Removes an entire mapbank from the project. 
        
        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        """
        bank = _canonical_form(bank)
        if bank in self.headers:
            del self.headers[bank]
            self.autosave()
        else:
            raise RuntimeError(f'Bank {bank} not existent.')

    def remove_header(self, bank, map_idx):
        """ Removes a map header from the project. 
        
        Parameters:
        -----------
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.
        """
        bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
        if bank in self.headers and map_idx in self.headers[bank]:
            del self.headers[bank][map_idx]
            self.autosave()
        else:
            raise RuntimeError(f'Header [{bank}, {map_idx.zfill(2)}] not existent.')
    
    def save_header(self, header, bank, map_idx):
        """ Saves a header.
        
        Parameters:
        -----------
        header : dict
            The header to save.
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
        if bank in self.headers and map_idx in self.headers[bank]:
            label, path, namespace = self.headers[bank][map_idx]
            with open(Path(path), 'w+', encoding=self.config['json']['encoding']) as f:
                json.dump({
                    'data' : header,
                    'label' : label,
                    'type' : self.config['pymap']['header']['datatype'],
                }, f, indent=self.config['json']['indent'])
        else:
            raise RuntimeError(f'No header located at [{bank}, {map_idx}]')

    def new_header(self, label, path, namespace, bank, map_idx):
        """ Creates a new header, assigns the namespace and saves it into a file.
        
        Parameters:
        -----------
        label : str
            The label of the header.
        path : str
            Path to the header structure file.
        namespace : str
            The namespace of the header.
        bank : int or str
            The map bank. If it is an integer or integer string, it is converted into its "canonical" form.
        map_idx : int or str
            The map index in the bank. If it is an integer or integer string, it is converted into its "canonical" form.

        Returns:
        --------
        header : dict
            The new header.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        bank, map_idx = _canonical_form(bank), _canonical_form(map_idx)
        if bank in self.headers:
            if map_idx in self.headers[bank]:
                raise RuntimeError(f'Index {map_idx} already present in bank {bank}')
            else:
                self.headers[bank][map_idx] = [label, os.path.relpath(path), namespace]
                datatype = self.config['pymap']['header']['datatype']
                header = self.model[datatype](self, [], [])
                # Assign the proper namespace
                set_member_by_path(header, namespace, self.config['pymap']['header']['namespace_path'])
                # Save the header
                self.save_header(header, bank, map_idx)
                self.autosave()
                return header
        else:
            raise RuntimeError(f'Bank {bank} not existent')

    def refactor_footer(self, label_old, label_new):
        """ Changes the label of a footer. Applies changes to all headers refering to this footer.
        
        Parameters:
        -----------
        label_old : str
            The label of the footer to change.
        label_new : str
            The new label of the footer. 
        """
        assert(label_old in self.footers), f'Footer {label_old} not existent.'
        assert(label_new not in self.footers), f'Footer label {label_new} already existent.'
        for bank in self.headers:
            for map_idx in self.headers[bank]:
                header, _, _ = self.load_header(bank, map_idx)
                if get_member_by_path(header, self.config['pymap']['header']['footer_path']) == label_old:
                    set_member_by_path(header, label_new, self.config['pymap']['header']['footer_path'])
                    self.save_header(header, bank, map_idx)
                    print(f'Refactored footer reference in header [{bank}, {map_idx.zfill(2)}]')
        footer, _ = self.load_footer(label_old)
        self.footers[label_new] = self.footers.pop(label_old)
        self.save_footer(footer, label_new)
        self.autosave()

    def remove_footer(self, label):
        """ Removes a footer form the project.
        
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

    def load_footer(self, label):
        """ Opens a footer by its label and verifies the label and type of json instance. 
        
        Parameters:
        -----------
        label : str
            The label of the footer to load.
        
        Returns:
        --------
        footer : dict or None
            The footer instance if present.
        footer_idx : int
            The index of the footer or -1 if no footer is present.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if label in self.footers:
            footer_idx, path = self.footers[label]
            with open(Path(path), encoding=self.config['json']['encoding']) as f:
                footer = json.load(f)
            assert(footer['label'] == label), 'Footer label mismatches the label stored in the project.'
            assert(footer['type'] == self.config['pymap']['footer']['datatype']), 'Footer datatype mismatches the configuration'
            return (footer['data'], footer_idx)
        else:
            return None, -1

    def save_footer(self, footer, label):
        """ Saves a footer. 
        
        Parameters:
        -----------
        footer : dict
            The footer to save.
        label : str
            The label of the footer.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if label in self.footers:
            footer_idx, path = self.footers[label]
            with open(Path(path), 'w+', encoding=self.config['json']['encoding']) as f:
                json.dump({
                    'data' : footer,
                    'label' : label,
                    'type' : self.config['pymap']['footer']['datatype'],
                }, f, indent=self.config['json']['indent'])
        else:
            raise RuntimeError(f'No footer {label}')

    def new_footer(self, label, path, footer_idx):
        """ Creates a new footer.
        
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
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if label in self.footers:
            raise RuntimeError(f'Footer {label} already present.')
        elif footer_idx not in self.unused_footer_idx():
            raise RuntimeError(f'Footer index {footer_idx} already present.')
        else:
            self.footers[label] = footer_idx, os.path.relpath(path)
            datatype = self.config['pymap']['footer']['datatype']
            footer = self.model[datatype](self, [], [])
            # Save the footer
            self.save_footer(footer, label)
            self.autosave()
            return footer

    def import_footer(self, label, path, footer_idx):
        """ Imports a new footer. 
        
        Parameters:
        -----------
        label : str
            The label of the footer. The json file will be modified s.t. the labels match.
        path : str
            Path to the footer file.
        footer_idx : int
            Index of the footer.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        if label in self.footers:
            raise RuntimeError(f'Footer {label} already existent.')
        if footer_idx not in self.unused_footer_idx():
            raise RuntimeError(f'Footer index {footer_idx} already in use.')
        with open(Path(path), encoding=self.config['json']['encoding']) as f:
            footer = json.load(f)
        assert(footer['type'] == self.config['pymap']['footer']['datatype']), 'Footer datatype mismatches the configuration'
        self.footers[label] = footer_idx, os.path.relpath(path)
        self.save_footer(footer['data'], label)
        self.autosave()

    def remove_tileset(self, primary, label):
        """ Removes a tileset from the project.
        
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

    def load_tileset(self, primary, label):
        """ Loads a tileset by its label and verifies label and type of the json instance. 
        
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
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        path = (self.tilesets_primary if primary else self.tilesets_secondary).get(label, None)
        if path is None:
            return None
        else:
            with open(Path(path), encoding=self.config['json']['encoding']) as f:
                tileset = json.load(f)
            assert(tileset['label'] == label), 'Tileset label mismatches the label stored in the project'
            assert(tileset['type'] == self.config['pymap'][('tileset_primary' if primary else 'tileset_secondary')]['datatype']), 'Tileset datatype mismatches the configuration'
            return tileset['data']

    def save_tileset(self, primary, tileset, label):
        """ Saves a tileset. 
        
        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset.
        tileset : dict
            The tileset structure.
        label : str
            The label of the tileset.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        if label in tilesets:
            path = tilesets[label]
            with open(Path(path), 'w+', encoding=self.config['json']['encoding']) as f:
                json.dump({
                    'data' : tileset,
                    'label' : label,
                    'type' : self.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['datatype'],
                }, f, indent=self.config['json']['indent'])
        else:
            raise RuntimeError(f'No tileset {label}')

    def new_tileset(self, primary, label, path, gfx_compressed=True, tileset=None):
        """ Creates a new tileset.
        
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
            The new tileset. If not given, an empty default tileset is created from the data model.
        """ 
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        if label in tilesets:
            raise RuntimeError(f'Tileset {label} already present.')
        else:
            tilesets[label] = os.path.relpath(path)
            config = self.config['pymap']['tileset_primary' if primary else 'tileset_secondary']
            if tileset is None:
                datatype = config['datatype']
                tileset = self.model[datatype](self, [], [])
                set_member_by_path(tileset, str(int(gfx_compressed)), config['gfx_compressed_path'])
            # Save the tileset
            self.save_tileset(primary, tileset, label)
            self.autosave()
            return tileset

    def refactor_tileset(self, primary, label_old, label_new):
        """ Changes the label of a tileset. Applies changes to all footers refering to this tileset. 
        
        Parameters:
        -----------
        primary : bool
            If the tileset is a primary tileset or not (secondary tileset).
        label_old : str
            The old label of the tileset to refactor.
        label_new : str
            Its new label.
        """
        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        assert(label_old in tilesets), f'Tileset {label_old} not existent.'
        assert(label_new not in tilesets), f'Tileset {label_new} already existent.'
        for label in self.footers:
            footer, _ = self.load_footer(label)
            if get_member_by_path(footer, self.config['pymap']['footer']['tileset_primary_path' if primary else 'tileset_secondary_path']) == label_old:
                set_member_by_path(footer, label_new, self.config['pymap']['footer']['tileset_primary_path' if primary else 'tileset_secondary_path'])
                self.save_footer(footer, label)
                print(f'Refactored tileset reference in footer {label}')
        tileset = self.load_tileset(primary, label_old)
        tilesets[label_new] = tilesets.pop(label_old)
        self.save_tileset(primary, tileset, label_new)
        self.autosave()

    def import_tileset(self, primary, label, path):
        """ Imports a tileset into the project. 
        
        Parameters:
        -----------
        primary : bool
            If the tileset will be a primary tileset.
        label : str
            The label of the tileset.
        path : str
            The path to the tileset.
        """
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        tilesets = self.tilesets_primary if primary else self.tilesets_secondary
        if label in tilesets:
            raise RuntimeError(f'Tileset {label} already existent.')
        with open(Path(path), encoding=self.config['json']['encoding']) as f:
            tileset = json.load(f)
        assert(tileset['type'] == self.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['datatype']), 'Tileset datatype mismatches the configuration'
        tilesets[label] = os.path.relpath(path)
        self.save_tileset(primary, tileset['data'], label)
        self.autosave()

    def load_gfx(self, primary, label):
        """ Loads a gfx and instanciates an agb image. 
        
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
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        gfx = self.gfxs_primary if primary else self.gfxs_secondary
        if label not in gfx:
            raise RuntimeError(f'No gfx associated with label {gfx}')
        else:
            path = gfx[label]
            img, _ = image.from_file(path)
            return img

    def save_gfx(self, primary, image, palette, label):
        """ Saves a gfx with a certain palette. 
        
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
        os.chdir(os.path.abspath(os.path.dirname(self.path)))
        gfx = self.gfxs_primary if primary else self.gfxs_secondary
        if label not in gfx:
            raise RuntimeError(f'No gfx associated with label {gfx}')
        else:
            path = gfx[label]
            image.save(path, palette)
    
    def refactor_gfx(self, primary, label_old, label_new):
        """ Changes the label of a gfx. Applies changes to all tilesets refering to this gfx. 
        
        Parameters:
        -----------
        primary : bool
            If the tileset is a primary gfx or not (secondary gfx).
        label_old : str
            The old label of the gfx to refactor.
        label_new : str
            Its new label.
        """
        gfxs = self.gfxs_primary if primary else self.gfxs_secondary
        assert(label_old in gfxs), f'Gfx {label_old} not existent.'
        assert(label_new not in gfxs), f'Gfx {label_new} already existent.'
        for label in (self.tilesets_primary if primary else self.tilesets_secondary):
            tileset = self.load_tileset(primary, label)
            if get_member_by_path(tileset, self.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['gfx_path']) == label_old:
                set_member_by_path(tileset, label_new, self.config['pymap']['tileset_primary' if primary else 'tileset_secondary']['gfx_path'])
                self.save_tileset(primary, tileset, label)
                print(f'Refactored gfx reference in tileset {label}')
        gfxs[label_new] = gfxs.pop(label_old)
        self.autosave()

    def remove_gfx(self, primary, label):
        """ Removes a gfx from the project.
        
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

    def import_gfx(self, primary, label, path):
        """ Imports a gfx into the project. Assertions on the bitdepth and image size are 
        performed.
        
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
            img, _ = agb.image.from_file(path)
            assert(img.depth == 4)
            assert(img.width % 8 == 0)
            assert(img.height % 8 == 0)
            if primary:
                assert(img.width * img.height == 320 * 128)
            else:
                assert(img.width * img.height == 192 * 128)
            gfxs[label] = os.path.relpath(path)
            self.autosave()


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
        if not namespace_constants is None:
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

def _canonical_form(x):
    try: return str(int(str(x), 0))
    except ValueError: return x