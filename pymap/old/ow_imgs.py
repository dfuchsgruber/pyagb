""" Class for lazy storage and access to overworld images 
from both extern .png references and intern references from a rom """

import PIL.Image as PImage
import PIL.ImageTk as ImageTk
import tempfile
from . import image as pyimage
import agb.img_dump
import agb.agbrom
import os

# Find the img none png
dir, _ = os.path.split(__file__)
IMG_NONE_PATH = os.path.join(dir, "pymap_img_assocs_none.png")

class Ow_imgs():
    
    def __init__(self, assocs_path, proj):
        """ Class to handle all overworld images in a lazy manner. It allows for links
        against a static rom and static pngs and otherwise displays a standard
        image for every overworld image. """
        self.proj = proj
        with open(assocs_path, "r") as f:
            d = eval(f.read())
        self.extern = d["extern"]
        rompath, table_off, number_range, pal_table = d["base"]
        if rompath: self.rom = agb.agbrom.Agbrom(self.proj.realpath(rompath))
        else: self.rom = None
        self.table = table_off
        if number_range: self.number_range = number_range
        else: self.number_range = set()
        self.pictures = [None for i in range(256)]
        self.pal_table = pal_table

    def get(self, i):
        """ Gets a tuple (pil image, photo image) of a picture """
        if i not in range(256): i = 0
        if not self.pictures[i]:

            # Compute picture either from rom or extern file
            if i in self.extern:
                # Load from extern file
                path = self.proj.realpath(self.extern[i])
                image = pyimage.Image()
                image.load_image_file(path)
            elif i in self.number_range:
                # Extract picture from rom
                ow_sprite = self.rom.pointer(self.table + 4 * i)
                graphics = self.rom.pointer(ow_sprite + 28)
                img_off = self.rom.pointer(graphics)
                width = self.rom.u16(ow_sprite + 8)
                height = self.rom.u16(ow_sprite + 10)
                pal_tag = self.rom.u16(ow_sprite + 2)
                pal_offset = self._pal_offset(pal_tag)
                # Dump picture to tempfile
                fd = tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False)
                path = fd.name
                agb.img_dump.dump_png_fp(fd, self.rom, img_off, width, height, pal_offset, 16, img_lz77=False, pal_lz77=False, depth=4)
                fd.close()
                # Load as pymap image
                image = pyimage.Image()
                image.load_image_file(path)
                # Delete tempfile
                os.remove(path)
            else:
                # From default none image
                image = pyimage.Image()
                image.load_image_file(IMG_NONE_PATH)

            width, height = image.width, image.height
            rawimage = image.get_image(int(width / 8), int(height / 8), image.palette)
            pimage = PhotoImage = ImageTk.PhotoImage(rawimage)
            self.pictures[i] = (rawimage, pimage)
        return self.pictures[i]
            
    def _pal_offset(self, tag):
        """ Returns proper pal offset by tag """
        off = self.pal_table
        while True:
            itag = self.rom.u16(off + 4)
            if itag == 0x11FF:
                raise Exception("Overworld is not associated with palette")
            if itag == tag:
                return self.rom.pointer(off)
            off += 8