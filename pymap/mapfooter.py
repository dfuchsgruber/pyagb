import json
from . import tileset
import tempfile


class Mapfooter:
    """ Class to model a mapfooter """
    def __init__(self):
        self.width, self.height = 1, 1
        self.borders = [[0]]
        self.tsp = None
        self.tss = None
        self.tsp_sym = None
        self.tss_sym = None
        self.border_width, self.border_height = 1, 1
        self.blocks = [[0]]
        self.padding = 0
        
    def to_dict(self):
        """ Returns a json exportable dict for all attributes """
        d = {}
        d["size"] = self.width, self.height
        d["borders"] = self.borders
        d["tsp"] = None if not self.tsp else self.tsp.symbol
        d["tss"] = None if not self.tss else self.tss.symbol
        d["border_size"] = self.border_width, self.border_height
        d["blocks"] = self.blocks
        return d
    

def from_dict(d, proj, instanciate_ts=True):
    """ Initializes an instance of Mapfooter from a dict d"""
    #Initialize standard mapfooter
    m = Mapfooter()
    m.width, m.height = d["size"]
    m.borders = d["borders"]
    m.blocks = d["blocks"]
    if d["tsp"]:
        if instanciate_ts: m.tsp = proj.get_tileset(d["tsp"])
        else: m.tsp = tileset.Tileset(True, symbol=d["tsp"])
        m.tsp_sym = d["tsp"]
    else: m.tsp = None
    if d["tss"]:
        if instanciate_ts: m.tss = proj.get_tileset(d["tss"])
        else: m.tss = tileset.Tileset(False, symbol=d["tss"])
        m.tss_sym = d["tss"]
    else: m.tss = None
    m.border_width, m.border_height = d["border_size"]
    return m


