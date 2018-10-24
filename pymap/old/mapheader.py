from . import mapfooter, mapevent, mapconnection
import json
import os
from . import path as rpath
import PIL.Image as PImage

PATH_UNSAFED = "Unsafed map"

def block_data(block):
    """ Returns a block id and level data"""
    return block & 0x3FF, ((block >> 10) & 0x3F)

class Mapheader:
    """ Class to model a mapheader """
    def __init__(self):
        self.footer = mapfooter.Mapfooter()
        self.persons = []
        self.warps = []
        self.signposts = []
        self.triggers = []
        self.levelscript_header = "0"
        self.connections = []
        self.music = 0
        self.id = 0
        self.name_bank = 0
        self.flash_type = 0
        self.weather = 0
        self.type = 0
        self.field_18 = 0
        self.field_1a = 0
        self.show_name = 0
        self.battle_style = 0
        self.key = PATH_UNSAFED
        self.symbol = "_unnamed_"

    def save(self, path):
        """ Serialization method for the mapheader """
        self.key = path
        d = self.__dict__.copy()
        d["footer"] = self.footer.to_dict()
        d["persons"] = [p.to_dict() for p in self.persons]
        d["warps"] = [w.to_dict() for w in self.warps]
        d["signposts"] = [s.to_dict() for s in self.signposts]
        d["triggers"] = [t.to_dict() for t in self.triggers]
        d["connections"] = [c.to_dict() for c in self.connections]
        fd = open(path, "w+")
        json.dump(d, fd)
        fd.close()

    def get_map_image(self):
        """ Returns a PIL image for this map (if possible, else returns None)"""
        if not self.footer: return None
        if not self.footer.tsp or not self.footer.tss: return None
        if self.footer.tsp.image.empty or self.footer.tss.image.empty: return None
        #First compute the block pool
        block_imgs = []
        for i in range(len(self.footer.tsp.blocks) + len(self.footer.tss.blocks)):
            if i < len(self.footer.tsp.blocks): data = self.footer.tsp.blocks[i]
            else: data = self.footer.tss.blocks[i - len(self.footer.tsp.blocks)]
            #Data is collected, now compute the block image
            img = PImage.new("RGBA", (16, 16), "black")
            for k in range(8):
                map_data = data[k]
                tile_id, vflip, hflip = map_data & 0x3FF, bool(map_data & 0x400), bool(map_data & 0x800)
                pal_id = (map_data >> 12) & 0xFF
                if pal_id > 12: continue
                if pal_id < 7: colors = self.footer.tsp.palettes[pal_id]
                else: colors = self.footer.tss.palettes[pal_id - 7]
                j = k & 3 #Index of inner loop (the computing loop is 2 layered, j runs from 0 to 3 for every layer)
                x, y = j & 1, j >> 1
                if tile_id < 0x280: tile_img =  self.footer.tsp.get_tile(tile_id, colors, hflip, vflip)
                else: tile_img =  self.footer.tss.get_tile(tile_id - 0x280, colors, hflip, vflip)
                if k < 4: #Paste without alpha channel on color #0
                    img.paste(tile_img, (8 * x, 8 * y))
                else: #Paste with alpha channel on color #0
                    img.paste(tile_img, (8 * x, 8 * y), tile_img)
            block_imgs.append(img)
        
        map_img = PImage.new("RGBA", (self.footer.width * 16, self.footer.height * 16), "black")
        for y in range(self.footer.height):
            for x in range(self.footer.width):
                block_id = self.footer.blocks[y][x] & 0x3FF
                map_img.paste(block_imgs[block_id], (x * 16, y * 16))
        return map_img






def load(path, proj, instanciate_ts=True):
    """ Instanciates a mapheader from a json file"""
    m = Mapheader()
    fd = open(path, "r+")
    d = json.load(fd)
    fd.close()
    for k, v in d.items(): m.__setattr__(k, v)
    m.footer = mapfooter.from_dict(m.footer, proj, instanciate_ts=instanciate_ts)
    m.persons = [mapevent.from_dict(pd) for pd in m.persons]
    m.warps = [mapevent.from_dict(wd) for wd in m.warps]
    m.signposts = [mapevent.from_dict(sd) for sd in m.signposts]
    m.triggers = [mapevent.from_dict(td) for td in m.triggers]
    m.connections = [mapconnection.from_dict(cd) for cd in m.connections]
    m.key = path
    #print(d, m.footer)
    return m

def references(path):
    """ Only retrieves references to tilesets and the namespace but does not instanciate the map at all (for time saving)"""
    fd = open(path, "r+")
    d = json.load(fd)
    fd.close()
    tsp = d["footer"]["tsp"]
    tss = d["footer"]["tss"]
    name_bank = d["name_bank"]
    return {
        "tsp" : tsp,
        "tss" : tss,
        "name_bank" : name_bank
    }    