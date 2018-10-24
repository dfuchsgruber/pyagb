""" Module that implements a virtual block """

class Block:

    def __init__(self):
        self.tiles = [None for i in range(0, 8)]


class Tile:
    """ Class that implements the reference to a
    tile in a .png file """

    def __init__(self, img_file, tile_id):
        self.img_file = img_file
        self.tile_id = tile_id