""" Class that stores gfx tile data of an image (png) file """

import png
import PIL.Image as PImage
from . import agbimg
import os.path

class Image:
    
    def __init__(self):
        self.tiles = None
        self.palette = None
        self.empty = True
        self.width, self.height = None, None
       

    def load_image_file(self, path):
        if not path:
            raise Exception("No path provided to load image")
        try: self.image = PImage.open(path)
        except Exception as e:
            print("Image", path, "could not be opened", e)
            return
        self.width, self.height = self.image.size
        #image = self._make_color_transparent(image)
        #self.tiles = self._img_to_tiles(image)
        self.palette = self.image.palette.palette
        self.empty = False
        self.tiles = self._img_to_tiles(self.image)



    def _img_to_tiles(self, image):
        width, height = image.size
        tiles = []
        #Splitting the image into tiles
        for y in range(0, height >> 3):
            for x in range(0, width >> 3):
                box = (x * 8, y * 8, min((x + 1) * 8, width), min((y + 1) * 8, height))
                tiles.append(image.crop(box))
        return tiles

    def _make_color_transparent(self, image, color=None):
        if not color:
            colors = image.palette.palette
            color = (colors[0], colors[1], colors[2])
        temp_image = image.convert("RGBA")
        data = [(color[0], color[1], color[2], 0) if pixel[0] == color[0] and pixel[1] == color[1] and pixel[2] == color[2] else pixel for pixel in temp_image.getdata()]
        temp_image.putdata(data)
        return temp_image

    def _tile_to_str(self, tile):
        str = ""
        for pixel in tile.getdata():
            str += hex(pixel[0]) + hex(pixel[1]) + hex(pixel[2])
        return str

    def _show_tiles(self, palette=None):
        tiles = self.tiles
        joined = PImage.new("RGBA", (len(tiles) * 8, 8))
        for i in range(0, len(tiles)):
            joined.paste(tiles[i], (i*8, 0))
        joined.show()

    def get_image(self, tiles_per_line, tiles_per_row, palette):
        img = self.image
        if self.image.size != (tiles_per_line * 8, tiles_per_row * 8):
            print("Warning - reshaping of tileset image causes delay...")
            img = self._get_tiles(tiles_per_line, tiles_per_row, palette)
        img.putpalette(palette)
        transparent_color = (palette[0], palette[1], palette[2])
        return self._make_color_transparent(img, color=transparent_color)

    def _get_tiles(self, tiles_per_line, tiles_per_row, palette):
        tiles = self.tiles
        joined = PImage.new("P", (tiles_per_line * 8, tiles_per_row * 8))
        for i in range(0, len(tiles)):
            x = i % tiles_per_line
            y = int(i / tiles_per_line)
            tiles[i].putpalette(palette)
            joined.paste(tiles[i], (x*8, y*8))
        return joined
        

    def get_tile(self, tile_id, palette, hflip, vflip):
        """ Gets a certain tile in a certain palette (used for block computing)"""
        tile = self.tiles[tile_id]
        tile.putpalette(palette)
        if hflip: tile = tile.transpose(PImage.FLIP_TOP_BOTTOM)
        if vflip: tile = tile.transpose(PImage.FLIP_LEFT_RIGHT)
        transparent_color = (palette[0], palette[1], palette[2])
        return self._make_color_transparent(tile, color=transparent_color)




