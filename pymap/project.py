from . import mapheader
from . import tileset
from . import image
import json
import os

""" Module that ressembles a project structure (maps, tilesets, etc. )"""

class Project:
    
    def __init__(self):
        
        self.banks = {}
        self.tilesets = {}
        self.images = {}
        self.path = None
        #self.project_root = "."

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
            return path
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
        path = self._sanitize(path)
        if not bank in self.banks: self.banks[bank] = {}
        self.banks[bank][map] = mh.symbol, path, namespace, footer_id
        mh.save(path)

    def get_tileset_paths(self, _sorted=True):
        """ Returns a list of all tileset paths """
        if _sorted: return list(sorted(self.tilesets.keys()))
        else: return list(self.tilesets.keys())

    def get_tileset(self, symbol, instanciate_image=True):
        """ Returns a tileset by its symbol 
        It therefore loads from path and creates a new instance that
        must be saved using self.save_tileset
        If instanciate_image is True then the image reference is
        resolved and the image applied to tileset. This might
        throw an Exception."""
        if symbol in self.tilesets:
            path = self.tilesets[symbol]
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
        """ Saves a tileset by its symbol and stores its symbol and path link in this project """
        path = self._sanitize(path)
        self.tilesets[symbol] = path
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
                    mh.save(path)
        path = self.tilesets.pop(symbol_old)
        self.tilesets[symbol_new] = path

        
    def get_image_paths(self, _sorted=True):
        """ Returns a list of all image paths """
        if _sorted: return list(sorted(self.images.keys()))
        return list(self.images.keys())

    def get_image_path(self, symbol):
        """ Returns an image object of a gfx symbol """
        if symbol in self.images:
            return self.images[symbol] 
        print("Warning - ", symbol, "is not associated with any .png file - return empty image")
        return None

    def save_image(self, symbol, path):
        """ Stores its symbol and path link in this project """
        self.images[symbol] = self._sanitize(path)
        
    def remove_image(self, symbol):
        """ Removes an image from the project and returns it at sucess and None at failue (e.g. image is not in this project)"""
        if symbol in self.images:
            return self.images.pop(symbol)
        return None

    def refractor_image_symbol(self, symbol_new, symbol_old):
        """ Refractors an image(gfx) symbol on all tilesets and thus
        may have (depending on the number of tilesets) a very
        high computation time"""
        if symbol_old not in self.images: raise Exception("Gfx symbol '" + symbol_old + "' is not defined!")
        for t_sym in self.tilesets:
           ts = self.get_tileset(t_sym, instanciate_image=False)
           if ts:
                if ts.gfx == symbol_old:
                    ts.gfx = symbol_new
                    ts.save(self.tilesets[sym])
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
        fd = open(path, "r+")
        dict = json.load(fd)
        fd.close()
        p = Project()
        for bank in dict["banks"]:
            p.banks[int(bank, 0)] = {}
            for mapid in dict["banks"][bank]:
                p.banks[int(bank, 0)][int(mapid, 0)] = dict["banks"][bank][mapid]
        p.tilesets = dict["tilesets"]
        p.images = dict["images"]
        p.path = path
        return p

    @staticmethod
    def _sanitize(path):
        return path.replace("\\", "/")

    @staticmethod
    def _sanitized_relative_path(path):
        return Project._sanitize(os.path.relpath(path))