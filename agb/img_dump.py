#!/usr/bin python3

from . import lz77
import png
import array

def img_to_tiles(data, tile_cnt, depth=4):
    """ Collects 8x8 Tiles from the img raw data """
    tiles = []
    filled = False
    for i in range(0, tile_cnt):
        #Loop to collect a tile
        tile_off = i * int(8 * 8 * depth / 8)
        tile = []
        for j in range(0, int(8 * 8 * depth / 8)):
            pixel_off = tile_off + j
            if depth == 4:
                try: 
                    tile.append(data[pixel_off] & 0xF)
                except:
                    tile.append(0)
                    filled = True
                try:
                    tile.append((data[pixel_off] >> 4) & 0xF)
                except:
                    tile.append(0)
                    filled = True
            elif depth == 8:
                try:
                    tile.append(data[pixel_off])
                except:
                    tile.append(0)
                    filled = True
            else: raise Exception("Unsupported bit-depth", depth)
        tiles.append(tile)
    if filled: print("Warning: Not enough tiles in img to create png of given size, filled with 0")
    return tiles

def tiles_to_rows(tiles, tile_width, tile_height):
    """ Builds png rows from a tile list"""
    rows = []
    for i in range(0, tile_height):
        tile_line_off = tile_width * i
        for j in range(0, 8):
            #Build 8 rows out of one tile line
            row = []
            for x in range(0, tile_width):
                #Iterate pixelwise through all the tiles
                tile = tiles[tile_line_off + x] #All Pixels inside the right tile
                pixels = tile[8 * j : 8 * (j+1)]
                row += pixels
            rows.append(row)
    return rows

def pal(data, color_cnt):
    """ Reads 16-Bit Palette entries from data and transforms it into an png rgb pal"""
    palette = []
    for i in range(0, color_cnt):
        gbapal = data[2 * i] | (data[2 * i + 1]) << 8
        red = gbapal & 31
        green = (gbapal >> 5) & 31
        blue = (gbapal >> 10) & 31
        palette.append((red * 8, green * 8, blue * 8))
    return palette


def dump_png_fp(fp, rom, img_offset, width, height, pal_offset, col_cnt, img_lz77=False, pal_lz77=False, depth=4, pal_start_color=0):
    """ Dumps a png with pal from data by fp """
    #First get picture
    img_data = rom.array(img_offset, (width * height * depth) >> 3)
    if img_lz77: img_data = lz77.decomp(rom, img_offset)
    tile_width = width >> 3
    tile_height = height >> 3
    img_tiles = img_to_tiles(img_data, tile_width * tile_height, depth=depth)
    img_rows = tiles_to_rows(img_tiles, tile_width, tile_height)

    #Now get the palette
    pal_data = rom.array(pal_offset, col_cnt * 2)
    if pal_lz77: pal_data = lz77.decomp(rom, pal_offset)
    pal_data = ([0,0] * pal_start_color) + pal_data
    img_pal = pal(pal_data, col_cnt+pal_start_color)
    
    pngwriter = png.Writer(width, height, palette=img_pal, bitdepth=depth)
    pngwriter.write(fp, img_rows)
    

def dump_png(path, rom, img_offset, width, height, pal_offset, col_cnt, img_lz77=False, pal_lz77=False, depth=4, pal_start_color=0):
    """ Dumps a png with pal from data by filename (creates fp and calls dump_png_fp) """
    with open(path, "wb") as fp:
        dump_png_fp(fp, rom, img_offset, width, height, pal_offset, col_cnt, img_lz77=img_lz77, pal_lz77=pal_lz77, depth=depth, pal_start_color=pal_start_color)

