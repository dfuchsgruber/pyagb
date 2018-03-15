#!/usr/bin/env python

""" Module to compile a .pmh pymap map header into a ARMv7 compilable assembly """
import sys
import getopt
import os
import time
import pymap.mapheader as mapheader
import pymap.mapevent as mapevent
import pymap.mapfooter as mapfooter
import pymap.constants as constants

def _flatten(l):
    return [item for sub in l for item in sub]

def shell(args):
    """ Transforms a map file (pmh) into an assembly file """
    #print("Pymap2s v0.1", os.getcwd())
    try:
        opts, args = getopt.getopt(args, "ho:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    outfile = None
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print("Usage: python pymap2s.py -o outfile infile projfile")
            return
        elif opt == "-o": outfile = os.path.abspath(arg)
    map_file = args[0]
    proj_file = args[1]

    # Initialize the constants module
    with open(proj_file + ".config", "r") as f:
        config = eval(f.read())
    const = constants.Constants(proj_file + ".constants", config["macro"])

    if not outfile: raise Exception("No output file specified")

    s = mapheader_to_assembly(map_file, const)
    fd = open(outfile, "w+")
    fd.write(s)
    fd.close()

def _indented(indent, s): return "".join(["\t"] * indent) + s

def _hexstr(i):
    try: return hex(i)
    except: return str(i)

def _data(indent, data):
    return "".join([_indented(indent, "." + type + " " + _hexstr(value) + "\n") for (value, type) in data])
        

def mapheader_to_assembly(map_file, const):
    """ Initializes a mapheader and exports it and all subcomponents """
    # Load the mapheader and omit instanciation of subcomponents
    # to improve performance
    maph = mapheader.load(map_file, None, instanciate_ts=False)

    # Create a preamble with all neccessary include directives
    preamble = """@*** Auto generated map assembly of {0} at {1} ***
    
    """.format(map_file, str(time.time()))
    for label in ("map_battle_styles", "map_namespaces", "map_flash_types",
        "songs", "map_types", "map_show_name_types", "map_weathers",
        "map_adjacency_types", "map_connections", "flags", "items", "vars",
        "person_behaviours"):
        preamble += const.get_include_directive(label, "as") + "\n"
    
    # Create the source code
    source = """.align 4
    .global {0}

    {0}:
    """.format(maph.symbol)

    source += _data(1, (
        (maph.symbol + "_footer", "word"), (maph.symbol + "_eventheader", "word"), (maph.levelscript_header, "word"),
        (maph.symbol + "_connectionheader", "word"), (maph.music, "hword"), (maph.id, "hword"), (maph.name_bank, "byte"), (maph.flash_type, "byte"),
        (maph.weather, "byte"), (maph.type, "byte"), (maph.field_18, "byte"), (maph.show_name, "byte"), (maph.field_1a, "byte"),
        (maph.battle_style, "byte") 
        )) + "\n\n"

    source += mapevents_to_assembly(maph)
    source += mapconnections_to_assembly(maph)
    source += mapfooter_to_assembly(maph)
    return preamble + source

def mapconnections_to_assembly(maph: mapheader.Mapheader):
    """ Dumps mapconnections to assembly content """
    s = ".align 4\n.global " + maph.symbol + "_connectionheader" + "\n" + maph.symbol + "_connectionheader:" + "\n"
    s += _indented(1, ".word " + str(len(maph.connections))) + "\n"
    if len(maph.connections):
        s += _indented(1, ".word " + maph.symbol + "_connections" + "\n\n")
        s += ".align 4\n.global " + maph.symbol + "_connections" + "\n" + maph.symbol + "_connections:" + "\n"
        for i in range(len(maph.connections)):
            s += _indented(1, "@// Connection number " + str(i) + "\n")
            s += _indented(1, ".word " + str(maph.connections[i].direction) + "\n")
            s += _indented(1, ".word " + str(maph.connections[i].displacement) + "\n")
            s += _indented(1, ".byte " + str(maph.connections[i].bank) + "\n")
            s += _indented(1, ".byte " + str(maph.connections[i].mapid) + "\n")
            s += _indented(1, ".byte " + str(maph.connections[i].fieldA) + "\n")
            s += _indented(1, ".byte " + str(maph.connections[i].fieldB) + "\n")
    else:
        s += _indented(1, ".word 0\n")
    return s + "\n"

def mapevents_to_assembly(maph: mapheader.Mapheader):
    """ Dumps mapevents to assembly content """
    s = ".align 4\n.global " + maph.symbol + "_eventheader" + "\n" + maph.symbol + "_eventheader:" + "\n"
    s += _indented(1, ".byte " + ", ".join(map(str, map(len, (maph.persons, maph.warps, maph.triggers, maph.signposts)))) + "\n")
    events = [maph.symbol + "_" + attr for attr in ("persons", "warps", "triggers", "signs")]
    if not len(maph.persons): events[0] = "0"
    if not len(maph.warps): events[1] = "0"
    if not len(maph.triggers): events[2] = "0"
    if not len(maph.signposts): events[3] = "0"

    s += _indented(1, ".word " + ", ".join(events) + "\n")
    s += "\n\n"

    if len(maph.persons):
        #Dump persons
        s += ".align 4\n.global " + maph.symbol + "_persons" + "\n" + maph.symbol + "_persons:\n"
        for i in range(len(maph.persons)):
            person = maph.persons[i]
            s += _indented(1, "@// Person #" + hex(i) + "\n")
            s += _data(1, (
                (person.target_index, "byte"), (person.picture, "byte"), (person.field_2, "byte"), (person.field_3, "byte"),
                (person.x, "hword"), (person.y, "hword"), (person.level, "byte"), (person.behaviour, "byte"),
                (person.behaviour_range, "byte"), (person.field_b, "byte"), (_hexstr(person.is_trainer) + " | ( " + _hexstr(person.is_trainer_padding) + " << 1 )", "byte" ),
                (person.field_d, "byte"), (person.alert_radius, "hword"), (person.script, "word"), (person.flag, "hword"), (person.field_16, "byte"),
                (person.field_17, "byte")
            ))
        s += "\n\n"

    if len(maph.warps):
        #Dump warps
        s += ".align 4\n.global " + maph.symbol + "_warps" + "\n" + maph.symbol + "_warps:\n"
        for i in range(len(maph.warps)):
            warp = maph.warps[i]
            s += _indented(1, "@// Warp #" + hex(i) + "\n")
            s += _data(1, (
                (warp.x, "hword"), (warp.y, "hword"), (warp.level, "byte"), (warp.target_warp, "byte"), (warp.target_map, "byte"),
                (warp.target_bank, "byte")
            ))
        s += "\n\n"
    
    if len(maph.triggers):
        #Dump triggers
        s += ".align 4\n.global " + maph.symbol + "_triggers" + "\n" + maph.symbol + "_triggers:\n"
        for i in range(len(maph.triggers)):
            trigger = maph.triggers[i]
            s += _indented(1, "@// Warp #" + hex(i) + "\n")
            s += _data(1, (
                (trigger.x, "hword"), (trigger.y, "hword"), (trigger.level, "byte"), (trigger.field_5, "byte"),
                (trigger.variable, "hword"), (trigger.value, "hword"), (trigger.field_a, "byte"), (trigger.field_b, "byte"),
                (trigger.script, "word")
            ))
        s += "\n\n"

    if len(maph.signposts):
        #Dump signs
        s += ".align 4\n.global " + maph.symbol + "_signs" + "\n" + maph.symbol + "_signs:\n"
        for i in range(len(maph.signposts)):
            sign = maph.signposts[i]
            s += _indented(1, "@// Warp #" + hex(i) + "\n")
            s += _data(1, (
                (sign.x, "hword"), (sign.y, "hword"), (sign.level, "byte"), (sign.sign_type, "byte"),
                (sign.field_6, "byte"), (sign.field_7, "byte")
            ))
            if sign.structure == mapevent.SIGN_STRUCTURE_SCRIPT:
                s += _data(1, [
                    (sign.script, "word")
                ])
            elif sign.structure == mapevent.SIGN_STRUCTURE_ITEM:
                s += _data(1,(
                    (sign.item_id, "hword"), (sign.hidden, "byte"), (sign.count, "byte")
                ))
            else: raise Exception("Unkown sign type " + sign.type)
        s += "\n\n"

    return s


def mapfooter_to_assembly(maph: mapheader.Mapheader):
    """ mapfooter -> assembly string """
    footer = maph.footer
    s = ".align 4\n.global " + maph.symbol + "_footer" + "\n" + maph.symbol + "_footer:" + "\n"
    s += _data(1, (
        (footer.width, "word"), (footer.height, "word"), (maph.symbol + "_border", "word"), (maph.symbol + "_blocks", "word"),
        (footer.tsp.symbol, "word"), (footer.tss.symbol, "word"),
        (footer.border_width, "byte"), (footer.border_height, "byte"), (footer.padding, "hword")
    ))
    s += "\n\n"

    s += ".align 4\n.global " + maph.symbol + "_border" + "\n" + maph.symbol + "_border:" + "\n"
    s += _data(1, (
        [(", ".join(map(hex, _flatten(maph.footer.borders))), "hword")]
    ))
    s += "\n\n"

    s += ".align 4\n.global " + maph.symbol + "_blocks" + "\n" + maph.symbol + "_blocks:" + "\n"
    s += _data(1, (
        [(", ".join(map(hex, _flatten(maph.footer.blocks))), "hword")]
    ))
    s += "\n\n"
    return s


if __name__ == "__main__":
    shell(sys.argv[1:])