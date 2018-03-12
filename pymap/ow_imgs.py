""" Class for lazy storage and access to overworld images 
from both extern .png references and intern references from a rom """

import PIL.Image as PImage
import PIL.ImageTk as ImageTk
import tempfile
from . import image as pyimage

class Ow_imgs():
    
    def __init__(self, img_dump_method, romclass, assocs, none_img):
        """ img_dump_method is derived from an inaccessible toplevel module, as is romclass"""
        fd = open(assocs, "r+")
        d = eval(fd.read())
        fd.close()
        self.extern = d["extern"]
        rompath, table_off, number_range, pal_table = d["base"]
        if rompath: self.rom = romclass(path=rompath)
        else: self.rom = None
        self.table = table_off
        if number_range: self.number_range = number_range
        else: self.number_range = set()
        self.pictures = [None for i in range(256)]
        self.pal_table = pal_table
        self.img_dump_method = img_dump_method
        self.none_img = none_img

    def get(self, i):
        """ Gets a tuple (pil image, photo image) of a picture """
        if i not in range(256): i = 0
        if not self.pictures[i]:
            #Compute the picture first

            if i in self.extern:
                #Image is an extern image reference
                imgpath = self.extern[i]
                fd = open(imgpath, "rb")
                image = PImage.open(fd)
                width, height = image.size
            else:
                #Create a temp file for image storage
                if i not in self.number_range:
                    fd = open(STDNONEPATH, "rb")
                    width, height = 16, 16
                else:
                    fd = tempfile.NamedTemporaryFile(mode="w+b")
                    ow_sprite = self.rom.pointer(self.table + 4 * i)
                    graphics = self.rom.pointer(ow_sprite + 28)
                    img_off = self.rom.pointer(graphics)
                    width = self.rom.u16(ow_sprite + 8)
                    height = self.rom.u16(ow_sprite + 10)
                    pal_tag = self.rom.u16(ow_sprite + 2)
                    pal_offset = self._pal_offset(pal_tag)
                    self.img_dump_method(fd, self.rom, img_off, width, height, pal_offset, 16, img_lz77=False, pal_lz77=False, depth=4)
            pymapimage = pyimage.Image()
            pymapimage.load_image_file(fd)
            image = pymapimage.get_image(int(width / 8), int(height / 8), pymapimage.palette)
            pimage = PhotoImage = ImageTk.PhotoImage(image)
            self.pictures[i] = (image, pimage)
            fd.close()
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