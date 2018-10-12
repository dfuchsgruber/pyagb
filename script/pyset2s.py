#!/usr/bin/env python
import json
import pymap.tileset
import sys
import time
import getopt
import os.path
from pymap import project

def _flatten(l):
    return [item for sub in l for item in sub]

def shell(args):
    """ Transforms a tileset file (pts) into an assembly file """
    try:
        opts, args = getopt.getopt(args, "ho:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    outfile = None
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print("Usage: python pyset2s.py -o outfile project infile")
            return
        elif opt == "-o": outfile = os.path.abspath(arg)
    proj_file = args[0]
    tileset_file = args[1]
    proj = project.Project.load_project(proj_file)
    if not outfile: raise Exception("No output file specified")
    s = tileset_to_assembly(tileset_file, proj)
    fd = open(outfile, "w+")
    fd.write(s)
    fd.close()

def tileset_to_assembly(tileset_file, proj):
    """ Transforms a tileset file into an assembly string """
    s = "@*** Auto generated tileset assembly of '" + tileset_file + "', " + str(time.time()) + "***\n\n"


    t = pymap.tileset.from_file(tileset_file, from_root=True, path_absolute=True) #This is executed from the root dir
    symbol = t.symbol
    includes = [proj.constants.get_include_directive(include, 'as') for include in proj.config['pyset2s']['constants']]

    #First create the header (global symbols)
    s += "\n".join(includes + [] + [".global " + symbol + g for g in ["", "_palettes", "_blocks", "_behaviours"]])
    s += "\n\n"

    #Create the tileset chunk
    s += ".align 4\n" + symbol + ":\n"
    tinfo = [1, 0, 0, 0] #We assume tileset is always compressed and pals are contained with no displacement (as they are compiled this way)
    s += ".byte " + ", ".join(map(str, tinfo)) + "\n"
    s += ".word " + t.gfx + "\n"
    s += ".word " + symbol + "_palettes\n"
    s += ".word " + symbol + "_blocks\n"
    s += ".word " + t.init_func + "\n"
    s += ".word " + symbol + "_behaviours\n"
    s += "\n\n"

    #Create the palettes chunk
    def pal_to_rgbs(pal):
        rgb = []
        for i in range(0, len(pal), 3):
            rgb.append((pal[i] >> 3) | (pal[i + 1] << 2) | (pal[i + 2] << 7))
        return rgb
    
    s += ".align 4\n" + symbol + "_palettes:\n"
    s += "\n".join([".hword " + ", ".join(map(str,pal_to_rgbs(palette))) for palette in t.palettes])
    s += "\n\n"

        
    #Create the blocks chunk
    s += ".align 4\n" + symbol + "_blocks:\n"
    s += ".hword " + ", ".join(list(map(str, _flatten(t.blocks)))) + "\n"
    s += "\n\n"

    #Create the behaviour chunk
    s += ".align 4\n" + symbol + "_behaviours:\n"
    s += ".word " + ", ".join(["( " + " | ".join(["(" + str(b) + " << " + str(s) + ")" for (b, s) in zip(behaviour, [0, 9, 14, 18, 24, 27, 29, 31])]) + ")" for behaviour in t.behaviours]) + "\n"
    s += "\n\n"


    return s



    
if __name__ == "__main__":
    shell(sys.argv[1:])