#!/usr/bin/python3

import png
import PIL.Image as Image

class Agb_image:
    """ Class to modell fast methods for 4bpp color index pictures"""

    def _flatten(self, l):
        return [item for sub in l for item in sub]

    def __init__(self):
        self.key = None
        self.palette = None
        self.empty = True
        self.width, self.height = None, None
        pass

    def load_png(self, path):
        """ Opens a 4bpp png """
        fd = open(path, "rb")
        self.width, self.height, map, attributes = png.Reader(fd).read()
        self.palette = attributes["palette"]
        self.pixels = [list(sub) for sub in map]
        self.flat_pixels = self._flatten(self.pixels)
        fd.close()

        self.key = path
        self.empty = False

    def new(self, size, palette):
        """ Creates a new, empty Agb_image of certain size and palette"""
        self.width, self.height = size
        self.palette = palette
        self.pixels = [[0] * self.width for i in range(self.height)]
        self.flat_pixels = self._flatten(self.pixels)
        self.empty = False


    def crop(self, box):
        """ Crops an area of the image where box is a quadruple of x0, y0, x1, y1 """
        x0, y0, x1, y1 = box
        pixels = [line[x0:x1] for line in self.pixels[y0:y1]]
        cropped = Agb_image()
        cropped.pixels = pixels
        cropped.width = x1 - x0
        cropped.height = y1 - y0

    def _paste_transparent(self, img, coords):
        """ Pastes a pixel image over the current image and treates 0 as transparent"""
        x, y = coords
        w, h = img.width, img.height
        for dy in range(h):
            for dx in range(w):
                pixel = img.pixels[dy][dx]
                if pixel:
                    self[x + dx][y + dy] = pixel

    def _paste_opaque(self, img, coords):
        """ Pastes one pixel image over the current one and does not treat index 0 als transparent"""
        x, y = coords
        w, h = img.width, img.height
        for dy in range(h):
            self.pixels[y + dy][x : x + dx] = img.pixels[dy]



    def paste(self, img, coords, transparent=True):
        """ Pastes one pixel image over the current and - if desired - treates color index 0 as transparent"""
        if transparent:
            self._paste_transparent(img, coords)
        else:
            self._paste_opaque(img, coords)

    def to_pil_img(self, palette):
        """ Returns a pil image out of the data"""
        img = Image.new("P", (self.width, self.height))
        img.putdata(self.flat_pixels)
        #img.putpalette(bytes(self._flatten(self.palette)))
        img.putpalette(palette)
        return img
        
    def get_image(self, tiles_per_line, tiles_per_row, palette):
        """ Excactly the same as to_pil_img, for interface reasons"""
        if tiles_per_line * 8 != self.width or tiles_per_row * 8 != self.height:
            raise Exception("Excepeted a tileset of measurements " + str(tiles_per_line * 8) + ", " + str(tiles_per_row * 8))
        return self.to_pil_img(palette)

    def overlay(self, other, position, box, channel=True, hflip=False, vflip=False):
        """ Special fast method to overlay an image over the instance where position
            marks the position to put it into, box defines the excert from the other
            image and if channel is set pal index 0 is treated as transparency
            hflip and vflip define if the image other is transformed before overlaying"""
        x, y = position
        x0, y0, x1, y1 = box
        dx = x1 - x0
        dy = y1 - y0
        for i in range(dy):
            #Iterate over all lines
            src_y = y1 - i - 1 if hflip else y0 + i
            for j in range(dx):
                #Iterate over every pixel
                src_x = x1 - j - 1 if vflip else x0 + j
                pixel = other.pixels[src_y][src_x]
                if pixel or not channel: #Only apply a pixel if it is not 0 (transparent) or channel is not active
                    self.pixels[y + dy][x + dx] = pixel
                    self.flat_pixels[self.width * (y + dy) + x + dx] = pixel
        #This is in O(dx * dy) which for tilemap creation sums over O(2*w*h)