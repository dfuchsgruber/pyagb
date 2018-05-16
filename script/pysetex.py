#!/usr/bin/env python
import agb.agbrom
import agb.img_dump
import sys
import getopt
import os
from pymap import tileset, palette, project, image
import json


def _flatten(l):
    return [item for sub in l for item in sub]

def _bitfield(val, lowest, highest):
    """ Extracts an integer bitfield out of a value from bit lowest (inclusive) to highest (exclusive)"""
    size = highest-lowest
    mask = (1 << (size)) - 1
    return (val >> lowest) & mask

def _mkdirs(dir):
    if not os.path.exists(dir):
        print("Creating directory {0}".format(dir))
        os.makedirs(dir)

def export_tileset(rom, proj: project.Project, offset, symbol, gfx_symbol, export_gfx=None, basename=None, forced_block_size=0, delete_anim_edit=False, mkdirs=False):
    """ Exports a tileset into a pts (pymap tileset file)"""
    
    print(hex(offset))
    tinfo = rom.array(offset, 4)
    compression = tinfo[0]
    is_primary = not (tinfo[1] & 1)
    gfx_offset = rom.pointer(offset + 4)
    pal_offset = rom.pointer(offset + 8)
    block_data = rom.pointer(offset + 12)
    anim_init = rom.u32(offset + 16)
    behaviour_offset = rom.pointer(offset + 20)
    
    t = tileset.Tileset(is_primary)
    if forced_block_size:
        t.blocks = [[0] * 8 for i in range(forced_block_size)]
        t.behaviours = [0] * forced_block_size
    dump_palettes(rom, pal_offset, t)
    if export_gfx:
        _mkdirs(os.path.dirname(export_gfx))
        dump_gfx(rom, gfx_offset, export_gfx, compression, pal_offset, is_primary)
        t.gfx = hex(gfx_offset + 0x08000000)
        proj = project.Project.load_project(proj.path)
        if mkdirs: _mkdirs(os.path.dirname(os.path.relpath(export_gfx)))
        proj.save_image(gfx_symbol if gfx_symbol else hex(gfx_offset + 0x08000000), os.path.relpath(export_gfx))
        proj.save_project(proj.path)
    if gfx_symbol:
        t.gfx = gfx_symbol

    #Initialize blocks
    block_cnt = len(t.blocks)
    for i in range(block_cnt):
        for j in range(8): t.blocks[i][j] = rom.u16(block_data + (i * 8 + j) * 2)
    
    #Initialize behaviours
    for i in range(block_cnt):
        value = rom.u32(behaviour_offset + i * 4)
        t.behaviours[i] = list(map(hex, [_bitfield(value, 0, 9), _bitfield(value, 9, 14), _bitfield(value, 14, 18),
                            _bitfield(value, 18, 24), _bitfield(value, 24, 27), _bitfield(value, 27, 29),
                            _bitfield(value, 29, 31), _bitfield(value, 31, 32)]))

        """
        #This exportation is only relevant for violet as the fourth bitfield (18:24) will be exported from (16:24) (battle bg)
        t.behaviours[i] = list(map(hex, [_bitfield(value, 0, 9), _bitfield(value, 9, 14), _bitfield(value, 14, 16),
                            _bitfield(value, 16, 24), _bitfield(value, 24, 27), _bitfield(value, 27, 29),
                            _bitfield(value, 29, 31), _bitfield(value, 31, 32)]))"""

    t.symbol = symbol
    if delete_anim_edit:
        t.init_func = "0"
    else:
        t.init_func = hex(anim_init)
    
    # Save the tileset
    # Reopen the project to prevent access violations

    path = os.path.relpath(basename + ".pts")
    proj = project.Project.load_project(proj.path) 
    if mkdirs: _mkdirs(os.path.dirname(basename))
    proj.save_tileset(t.symbol, t, path)
    proj.save_project(proj.path)


def dump_palettes(rom, offset, t: tileset.Tileset):
    """ Dumps a palette array of size 7 or 5, depending on wether the tileset is primary and attaches them to the tileset"""
    if not t.is_primary: offset += 0xE0
    for i in range((7 if t.is_primary else 6)):
        colors = [rom.u16(offset + 32 * i + 2 * j) for j in range(16)]
        rgbs = _flatten(map(lambda col: (col & 0x1F, (col >> 5) & 0x1F, (col >> 10) & 0x1F), colors))
        t.palettes[i] = list(map(lambda col: col * 8, rgbs))
        """
        DEPRECATED TO STORE PALETTES IN FILES (NOW THEY ARE BOUND TO TILESET)
        index = i if t.is_primary else i + 7
        path = basename + "_pal" + str(i) + ".ppl"
        fd = open(path, "wb")
        fd.write(bytes([rgb * 8 for rgb in rgbs]))
        fd.close()
        t.palettes[i].load_palette_file(path)
        """

def dump_gfx(rom, offset, path, lz77, palette_offset, is_primary):
    """ Dumps the gfx """
    width, height = (128, 320) if is_primary else (128, 192)
    agb.img_dump.dump_png(path, rom, offset, width, height, palette_offset, 16, img_lz77=lz77, pal_lz77=False)



def main(args):
    """ Shell interface for the exporter """
    try:
        opts, args = getopt.getopt(args, "hg:s:o:n:t:b:x:y:l:", ["help", "verbose", "deleteanim", "mkdirs"])
    except getopt.GetoptError:
        sys.exit(2)
    rom = None
    offset = None
    export_gfx = None
    basename = None
    tileset_table = None
    tileset_number = None
    symbol = None
    gfx_symbol = None
    forced_block_size = 0
    delete_anim = False
    mkdirs = False
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("""Usage: python pysetex.py [opts] srcfile projfile
            -g {file}\t\tOutput graphic file [default=do not export graphic]
            -s {off}\t\t\ttileset offset
            -n {num}\t\t\ttileset number (overrides tileset offset together with tileset table)
            -t {off}\t\t\ttileset table (only needed when tileset number is passed) [default=DEFAULT_TABLE=0x2D49B8]
            -o {basepath}\t\toutput base path 'e.g. ts/tileset_foo would provide files ts/tileset_foo.pts'
            -y {name}\t\t\tsymbol to create out of tileset [default=maptileset_{(offset-table) / 24 (=num)}]
            -x {sym}\t\t\tsymbol reference to tileset gfx [default=original gfx offset]
            -b {num}\t\t\texplicit number of blocks to export (recommended only for secondary tilesets)
            --deleteanim\t\tDeletes the animation function reference and overrides with 0
            --mkdirs\t\ŧCreates neccessary subdirectories
            """)
            return
        elif opt in ("-g"): export_gfx = arg
        elif opt in ("-s"): offset = int(arg, 0)
        elif opt in ("-n"): tileset_number = int(arg, 0)
        elif opt in ("-t"): tileset_table = int(arg, 0)
        elif opt in ("-o"): basename = arg
        elif opt in ("-y"): symbol = arg
        elif opt in ("-x"): gfx_symbol = arg
        elif opt in ("-b"): forced_block_size = int(arg, 0)
        elif opt in ("--deleteanim"): delete_anim = True
        elif opt in ("--mkdirs"): mkdirs = True
    
    # Process args
    try: rom = agb.agbrom.Agbrom(args[0])
    except: raise Exception("No input rom specified (see --help)")
    try: projfile = args[1]
    except: raise Exception("No project file specified (see --help)")

    # Load config
    with open(projfile + ".config", "r") as f:
        config = eval(f.read())["pysetex"]

    if not tileset_table:
        tileset_table = config["tileset_base"]

    # Process values
    if tileset_number != None and tileset_table != None: offset = tileset_table + 24 * tileset_number
    if not offset: raise Exception("No offset is specified or offset is NULL (see --help)")
    if not basename: raise Exception("No output basepath specified (see --help)")
    if not symbol:
        tileset_number = int((offset - tileset_table) / 24)
        symbol = config["default_symbol"].format(hex(offset), str(tileset_number))

    #Open project
    p = project.Project.load_project(projfile)

    export_tileset(rom, p,  offset, symbol, gfx_symbol, export_gfx, basename, forced_block_size, delete_anim, mkdirs=mkdirs)



if __name__ == "__main__":
    main(sys.argv[1:])
        


    