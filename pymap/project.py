#!/usr/bin/python3

from . import mapheader
from . import tileset
from . import image
from . import constants
from . import ow_imgs
import json
import os

""" Module that ressembles a project structure (maps, tilesets, etc. )"""

class Project:
    
    def __init__(self):
        
        self.banks = {}
        self.tilesets = {}
        self.images = {}
        self.path = None
        self.constants = None
        self.config = {}
        self.ow_img_pool = None

    def projpath(self, path):
        """ Creates a path realtive to the project file. """
        projdir = os.path.dirname(self.path)
        return os.path.relpath(path, start=projdir)

    def realpath(self, path):
        """ Retrieves the real path from a path that is
        relative to the projpath """
        return os.path.join(os.path.dirname(self.path), path)

    def get_smallest_availible_foooter_id(self):
        """ Returns the smallest availible footer id """
        used_footers = self.get_used_footers()
        if len(used_footers):
            for i in range(1, max(used_footers) + 2):
                if not i in used_footers: return i
            raise Exception("Appearantly there are no availible footers which is rather unlikley since there are 0x10000 possible ones...")
        else:
            return 0

    def get_used_footers(self):
        """ Returns all footers that are in use (as set) """
        used_footers = set()
        for bank in self.banks:
            for mapid in self.banks[bank]:
                _, _, _, footer_id = self.banks[bank][mapid]
                used_footers.add(footer_id)
        return used_footers

    def get_footer_usage(self, footer_id):
        """ Returns all map symbols that use a footer id"""
        symbols = set()
        for bank in self.banks:
            for mapid in self.banks[bank]:
                symbol, _, _, fid = self.banks[bank][mapid]
                if fid == footer_id: symbols.add(symbol)
        return symbols

    def get_map_path(self, bank, mapid):
        """ Returns the path of a map by its bank, map"""
        try:
            symbol, path, namespace, footer_id = self.banks[bank][mapid]
            return self.realpath(path)
        except: return None

    def get_map_location_by_symbol(self, symbol):
        """ Returns a tuple bank,mapid for a given symbol if present or None instead """
        for bank in self.banks:
            for mapid in self.banks[bank]:
                s, _, _, _ = self.banks[bank][mapid]
                if s == symbol: return (bank, mapid)
        return None

    def save_map(self, bank, map, mh: mapheader.Mapheader, path):
        """ Saves a map by its symbol and stores its symbol and path link in this project """
        footer_id = mh.id
        symbol = mh.symbol
        namespace = mh.name_bank
        projpath = self.projpath(self._sanitize(path))
        if not bank in self.banks: self.banks[bank] = {}
        self.banks[bank][map] = mh.symbol, projpath, namespace, footer_id
        mh.save(path)

    def get_tileset_path(self, symbol):
        """ Returns the path of a tileset (realpath) relative to cwd """
        if not symbol in self.tilesets:
            raise Exception("Tileset symbol {0} could not be resolved!".format(symbol))
        return self.realpath(self.tilesets[symbol])

    def get_tileset_paths(self, _sorted=True):
        """ Returns a list of all tileset paths relative to cwd"""
        paths = [self.get_tileset_path(symbol) for symbol in self.tilesets]
        if _sorted:
            return sorted(paths)
        else:
            return paths

    def get_tileset_symbols(self, _sorted=True):
        """ Returns all tileset symbols. """
        symbols = list(self.tilesets.keys())
        if _sorted: symbols.sort()
        return symbols


    def get_tileset(self, symbol, instanciate_image=True):
        """ Returns a tileset by its symbol 
        It therefore loads from path and creates a new instance that
        must be saved using self.save_tileset
        If instanciate_image is True then the image reference is
        resolved and the image applied to tileset. This might
        throw an Exception."""
        if symbol in self.tilesets:
            path = self.realpath(self.tilesets[symbol])
            try: ts = tileset.from_file(path)
            except Exception as e:
                print("Exception while loading tileset " + symbol + " at '" +  path + "': " + str(e))
                return None
            img_symbol = ts.gfx
            if instanciate_image:
                try: ts.load_image_file(self.get_image_path(img_symbol))
                except Exception as e: print("Warning - image file could not be loaded: " + str(e))
            return ts

    def save_tileset(self, symbol, t: tileset.Tileset, path):
        """ Saves a tileset by its symbol and stores its symbol and path link in this project.
        The path parameter must be realtive to cwd or absolute. """
        projpath = self.projpath(path)
        self.tilesets[symbol] = self._sanitize(projpath)
        t.path = path
        t.save(path)

    def remove_tileset(self, symbol):
        """ Removes a tileset link from this project and returns the path """
        if symbol in self.tilesets:
            return self.tilesets.pop(symbol)
        else: return None
        


    def refractor_tileset_symbol(self, symbol_new, symbol_old):
        """ Refractors a tileset symbol on all maps and thus
        may have (depending on the number of maps) a very
        high computation time"""
        if symbol_old not in self.tilesets: raise Exception("Tileset symbol '" + symbol_old + "' is not defined!")
        for bank in self.banks:
            for mapid in self.banks[bank]:
                _, path, _, _ = self.banks[bank][mapid]
                changed = False
                mh = mapheader.load(path, self, instanciate_ts=False)
                if mh.footer.tsp.symbol == symbol_old:
                    mh.footer.tsp.symbol = symbol_new
                    changed = True
                if mh.footer.tss.symbol == symbol_old:
                    mh.footer.tss.symbol = symbol_new
                    changed = True
                if changed:
                    mh.save(self.realpath(path))
        path = self.tilesets.pop(symbol_old)
        self.tilesets[symbol_new] = path

    def get_image_symbols(self, _sorted=True):
        """ Returns a list of all image symbols. """
        symbols = list(self.images.keys())
        if _sorted: symbols.sort()
        return symbols

    def get_image_paths(self, _sorted=True):
        """ Returns a list of all image paths (realpath relative to cwd)"""
        paths = [self.get_image_path(symbol) for symbol in self.images]
        if _sorted:
            return sorted(paths)
        else:
            return paths

    def get_image_path(self, symbol):
        """ Returns an image object of a gfx symbol (realpath relative to cwd)"""
        if symbol in self.images:
            return self.realpath(self.images[symbol])
        print("Warning - ", symbol, "is not associated with any .png file - return empty image")
        return None

    def save_image(self, symbol, path):
        """ Stores its symbol and path link in this project.
        The path parameter must be absolute or relative to cwd."""
        path = self.projpath(path)
        self.images[symbol] = self._sanitize(path)
        
    def remove_image(self, symbol):
        """ Removes an image from the project and returns it at success and None at failue (e.g. image is not in this project)"""
        if symbol in self.images:
            return self.images.pop(symbol)
        return None

    def refractor_image_symbol(self, symbol_new, symbol_old):
        """ Refractors an image(gfx) symbol on all tilesets and thus
        may have (depending on the number of tilesets) a very
        high computation time"""
        if symbol_old not in self.images: raise Exception("Gfx symbol '" + symbol_old + "' is not defined!")
        
        # Scan all tilesets for usage of gfx and resave
        # all tilesets that use symbol_old.
        for t_sym in self.tilesets:
           ts = self.get_tileset(t_sym, instanciate_image=False)
           if ts:
                if ts.gfx == symbol_old:
                    # Change the symbol in the corresponding tileset
                    ts.gfx = symbol_new
                    path = self.get_tileset_path(t_sym)
                    ts.save(path)
                    
        # Change intern path link (projpath)
        path = self.images.pop(symbol_old)
        self.images[symbol_new] = path

    def save_project(self, path=None, locked=False):
        if not path: path = self.path
        dict = {
            "banks" : self.banks,
            "tilesets" : self.tilesets,
            "images" : self.images,
        }
        fd = open(path, "w+")
        fd.write(json.dumps(dict))
        fd.close()

    @staticmethod
    def load_project(path):
        with open(path, "r") as f:
            dict = json.load(f)
        p = Project()
        for bank in dict["banks"]:
            p.banks[int(bank, 0)] = {}
            for mapid in dict["banks"][bank]:
                p.banks[int(bank, 0)][int(mapid, 0)] = dict["banks"][bank][mapid]
        p.tilesets = dict["tilesets"]
        p.images = dict["images"]
        p.path = path

        # Initialze meta data from files
        with open(path + ".config", "r") as f:
            macro_config = eval(f.read())["macro"]

        p.constants = constants.Constants(path + ".constants", macro_config)
        p.ow_img_pool = ow_imgs.Ow_imgs(path + ".owimgassocs", p)
        return p

    @staticmethod
    def _sanitize(path):
        return path.replace("\\", "/")

    @staticmethod
    def _sanitized_relative_path(path):
        return Project._sanitize(os.path.relpath(path))


# Find the project templates
dir, _ = os.path.split(__file__)
TEMPLATE_DIR = os.path.join(dir, "templates")

def create_project(path):
    """ Creates a new project and all neccessary auxilliary files """
    proj = Project()
    proj.save_project(path)
    print("Saved project to {0}".format(path))

    # Create a constants 
    const_templ_path = os.path.join(TEMPLATE_DIR, "constants")
    with open(const_templ_path, "r") as f:
        const_templ = f.read()
    const_path = path + ".constants"
    with open(const_path, "w+") as f:
        f.write(const_templ)
    print("Saved constants to {0}".format(const_path))

    # Create config
    conf_templ_path = os.path.join(TEMPLATE_DIR, "config")
    with open(conf_templ_path, "r") as f:
        conf_templ = f.read()
    conf_path = path + ".config"
    with open(conf_path, "w+") as f:
        f.write(conf_templ)
    print("Saved configurations to {0}".format(conf_path))

    # Create owimgassocs
    owimg_templ_path = os.path.join(TEMPLATE_DIR, "owimgassocs")
    with open(owimg_templ_path, "r") as f:
        owimg_templ = f.read()
    owimg_path = path + ".owimgassocs"
    with open(owimg_path, "w+") as f:
        f.write(owimg_templ)
    print("Saved overworld image associations to {0}".format(owimg_path))
