#!/usr/bin/env python3

""" This module provides functionality for exporting maps batchwise. """
import sys
import getopt
import os
import pymapex
from pymap import project
import agb.agbrom


def main(args):
    """ Shell interface """

    # Setup argparser
    try:
        opts, args = getopt.getopt(args, "ho:s:t:", ["help", "pedantic", "mkdirs"])
    except getopt.GetoptError:
        sys.exit(2)

    mkdirs = False
    pedantic = False
    table = None
    symbol = None
    basepath = None
    
    # Parse opts
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(""" Usage pymapbatchex.py [opts] rom proj batch
Exports a batch of maps from a given rom into a given project.
Options:
    --mkdirs        :   Creates necessary subdirectories
    --pedantic      :   Throws an exception if a value could not
                        be mapped to a constant
    -o {basepath}   :   Defines the output rule for map basepaths.
                        As in the config file {0} is replaced with
                        the bank, {1} with the mapid, {2} with the
                        offset. Overrides .config file.
    -s {symbol}     :   Defines the symbol that will be created.
                        E.g. for foo 'foo_footer' will be created.
                        As in the config file {0} is replaced with
                        the bank, {1} with the mapid, {2} with the
                        offset. Overrides .config file.
    -t {offset}     :   Map table offset that is indexed by bank,
                        mapid. [default from config]
    
The output path for maps,  are specified in the .config
file of the project. This batch exporter uses the pymapex module. To
define backend and other settings use the config file.

A batch file only contains a comma seperated list of entities of form
<bank>.<mapid>. There may also be multiple lines. Lines begining with
'#' will be ignored (comments).
            """)
            exit(0)
        elif opt in ("-o"):
            basepath = arg
        elif opt in ("-s"):
            symbol = arg
        elif opt in ("-t"):
            table = int(arg, 0)
        elif opt in ("--mkdirs"):
            mkdirs = True
        elif opt in ("--pedantic"):
            pedantic = True
    
    # Parse args
    try:
        rom = args[0]
    except:
        raise Exception("No rom specified (see --help).")
    try:
        proj = args[1]
    except:
        raise Exception("No project specified (see --help).")
    try:
        batch = args[2]
    except:
        raise Exception("No batch specified (see --help).")

    # Load configuration
    with open(proj + ".config", "r") as f:
        config = eval(f.read())["pymapex"]
    batch_config = config["batch"]
    if not basepath: basepath = batch_config["map_output_path"]
    if not symbol: symbol = batch_config["map_symbol"]
    if not table: table = config["maptable"]
    
    rom = agb.agbrom.Agbrom(rom)
    proj = project.Project.load_project(proj)
    maps = parse_batchfile(batch)
    batch_export(rom, proj, maps, basepath, symbol, table, config, pedantic=pedantic, mkdirs=mkdirs)

def batch_export(rom, proj, maps, basepath, symbol, map_table, config, pedantic=False, mkdirs=False):
    """ Performs batch export on a list of tuples bank, mapid """

    for bank, mapid in maps:
        offset = rom.pointer(rom.pointer(rom.pointer(map_table) + 4 * bank) + 4 * mapid)
        map_symbol = symbol.format(bank, mapid, offset)
        map_basepath = basepath.format(bank, mapid, offset)
        header = pymapex.export_map(rom, offset, None, None, map_symbol, map_basepath, 
        proj, config, pedantic=pedantic)
        if mkdirs: pymapex._mkdirs(os.path.dirname(map_basepath))
        proj = project.Project.load_project(proj.path)
        proj.save_map(bank, mapid, header, map_basepath + ".pmh")
        proj.save_project()
        print("Exported map {0}.{1} into {2} with symbol {3}.".format(str(bank), str(mapid),
        map_basepath, map_symbol))


def parse_batchfile(batchfile):
    """ Parses an input batchfile and returns a list of 
    tuples bank, mapid """
    with open(batchfile, "r") as f:
        lines = f.read().split("\n")
    maps = []
    
    # Parse linewise
    for line in lines:
        line = line.strip()
        if not len(line): continue
        if line.startswith("#"): continue
        tokens = line.split(",")
        for token in tokens:
            components = token.split(".")
            bank = int(components[0].strip())
            mapid = int(components[1].strip())
            maps.append((bank, mapid))
    
    return maps


if __name__ == "__main__":
    main(sys.argv[1:])