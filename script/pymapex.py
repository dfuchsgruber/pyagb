#!/usr/bin/env python
import agb.agbrom
import pymap
import sys
import getopt
import os
from pymap import mapfooter, mapheader, mapevent, tileset, project, mapconnection
import json

def _flatten(l):
    return [item for sub in l for item in sub]

def _mkdirs(dir):
    if not os.path.exists(dir):
        print("Creating directory {0}".format(dir))
        os.makedirs(dir)

def main(args):
    """ Shell interface for the exporter """
    try:
        opts, args = getopt.getopt(args, "hb:m:s:t:p:c:y:o:", ["help", "pedantic", "mkdirs"])
    except getopt.GetoptError:
        sys.exit(2)
    rom = None
    bank = None
    map = None
    offset = None
    table = None
    tsp = None
    tss = None
    basepath = None
    symbol = None
    pedantic = False
    mkdirs = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("""Usage: python pymapex.py [flags] rom project
            -o {path}\t\tOutput base path (without file extension)
            -b {bank}\t\tMap bank to export from
            -m {map}\t\tMap number inside bank to export
            -t {table}\t\tMapbank table offset
            -s {offset}\t(Optional) export mapheader from ceratin offset (overrides -b, -m and -t)
            -p {symbol}\tSymbol of primary tileset. If not defined the tileset will be exported
            -c {symbol}\tSymbol of secondary tileset. If not defined the tileset will be exported
            -y {symbol}\tBase symbol (e.g. if symbol equals 'foo' then global symbols for mapheader, footer, etc. will be 'foo', 'foo_footer', ..., etc.)
            --pedantic\t\tPedantic export for constants (if a value can not be constantized exporting will be aborted)
            --mkdirs\t\ŧCreates output directories if necessary
            """)
            return
        elif opt in ("-b"): bank = int(arg, 0)
        elif opt in ("-m"): map = int(arg, 0)
        elif opt in ("-t"): table = int(arg, 0)
        elif opt in ("-s"): offset = int(arg, 0)
        elif opt in ("-p"): tsp = arg
        elif opt in ("-c"): tss = arg
        elif opt in ("-y"): symbol = arg
        elif opt in ("-o"): basepath = arg
        elif opt in ("--pedantic"): pedantic = True
        elif opt in ("--mkdirs"): mkdirs = True

    try: rom = agb.agbrom.Agbrom(args[0])
    except: raise Exception("No input rom specified (see --help)")
    try: projfile = args[1]
    except: raise Exception("No project file specified (see --help)")
    if not basepath: raise Exception("No output path specified!")
    proj = project.Project.load_project(projfile)
    if not symbol: raise Exception("No symbol specified (see --help)")

    # Load the configuration for pymapex
    with open(projfile + ".config") as f:
        conf = eval(f.read())["pymapex"]

    # Retrieve the offset from bank, mapid tuple if not specified
    if offset == None:
        # Try to process the table
        if bank == None: raise Exception("No bank specified")
        if map == None: raise Exception("No map specified")
        if table == None: table = conf["maptable"]
        offset = rom.pointer(rom.pointer(rom.pointer(table) + 4 * bank) + 4 * map)

    # Export the map file and recursively invoke backends according
    # to the configuration.
    # Reopen the projects after invokes to prevent ereasing of
    # changes done by invokes.
    header = export_map(rom, offset, tsp, tss, symbol, basepath, proj, conf, pedantic=pedantic)
    if mkdirs: _mkdirs(os.path.dirname(basepath))
    proj = project.Project.load_project(proj.path)
    proj.save_map(bank, map, header, basepath + ".pmh")
    proj.save_project()

def export_connections(rom: agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports the map connections """
    count = rom.u32(offset)
    connections = [mapconnection.Mapconnection() for i in range(count)]
    base = rom.pointer(offset + 4)
    for i in range(count):
        offset = base + 12 * i
        direction = rom.u32(offset)
        connections[i].direction = proj.constants.constantize(rom.u32(offset), "map_connections", pedantic=pedantic)
        connections[i].displacement = rom._int(offset + 4)
        connections[i].bank = rom.u8(offset + 8)
        connections[i].mapid = rom.u8(offset + 9)
        connections[i].fieldA = rom.u8(offset + 10)
        connections[i].fieldB = rom.u8(offset + 11)
    return connections

def export_levelscript(rom: agb.agbrom.Agbrom, offset, type, basepath, proj, config, pedantic=False):
    """ Exports a levelscript structure as assembly string.
    Returns a tuple: label, assembly"""
    if type in (1,3,5,6,7):
        # Unextended format
        ow_script_config = config["backend"]["ow_script"]
        cmd = ow_script_config["exec"].format(rom.path, proj.path, hex(offset), basepath, "lscr")
        if os.system(cmd):
            raise Exception("Execution of backend script exported produced exit code != 0")
        label = ow_script_config["label"].format(rom.path, proj.path, hex(offset), basepath, "lscr")
        return label, ""
    elif type in (2,4):
        # Extended format
        ow_script_config = config["backend"]["ow_script"]
        label = "lscr_" + hex(offset)
        assembly = ".align 4\n.global " + label + "\n\n" + label + ":\n"
        var = proj.constants.constantize(rom.u16(offset), "vars", pedantic=pedantic)
        value = hex(rom.u16(offset + 2))
        script_offset = rom.nullable_pointer(offset + 4)
        cmd = ow_script_config["exec"].format(rom.path, proj.path, hex(script_offset), basepath, "lscr")
        if os.system(cmd):
            raise Exception("Execution of backend script exported produced exit code != 0")
        script_label = ow_script_config["label"].format(rom.path, proj.path, hex(script_offset), basepath, "lscr")
        field_8 = hex(rom.u16(offset + 8))
        assembly += "\t.hword " + var + ", " + value + "\n"
        assembly += "\t.word " + script_label + "\n"
        assembly += "\t.hword " + field_8 + "\n"
        return label, assembly
    else: raise Exception("Unkown levelscript header type " + str(type))

def export_levelscripts(rom: agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports levelscripts into a seperate folder """
    lscr = ".global lscr_" + hex(offset) + "\n\nlscr_" + hex(offset) + ":\n"
    lscr_label = "lscr_" + hex(offset)
    assemblies = []
    while True:
        type = rom.u8(offset)
        lscr += "\t.byte " + hex(type) + "\n"
        if not type: break
        label, assembly = export_levelscript(rom, rom.pointer(offset + 1), type, basepath, proj, config, pedantic=pedantic)
        offset += 5
        lscr += ".word " + label + "\n"
        assemblies.append(assembly)
    lscr += "\n\n".join([""] + assemblies)
    _mkdirs(os.path.dirname(basepath))
    filename = os.path.join(os.path.dirname(basepath), "lscr.asm")
    
    # Write output file
    preamble = ""
    for label in ("vars", "map_connections"):
        preamble += proj.constants.get_include_directive(label, "as") + "\n"
    with open(filename, "w+") as f:
        f.write(preamble + "\n\n" + lscr)
    return lscr_label

        

def export_map(rom, offset, tsp, tss, symbol, basepath, proj, config, pedantic=False):
    """ Exports a map """
    header = mapheader.Mapheader()
    export_footer(header.footer, rom, rom.pointer(offset), tsp, tss, basepath, proj, config, pedantic=pedantic)
    event_off = rom.u32(offset + 4)
    if event_off: export_events(header, rom, rom.pointer(offset + 4), basepath, proj, config, pedantic=pedantic)
    else: event_off = "0"
    lscr_off = rom.u32(offset + 0x8)
    if lscr_off: header.levelscript_header = export_levelscripts(rom, rom.pointer(offset + 0x8), basepath, proj, config, pedantic=pedantic)
    else: header.levelscript_header = "0"
    if rom.u32(offset + 0xC) == 0:
        header.connections = []
    else:
        header.connections = export_connections(rom, rom.pointer(offset + 0xC), basepath, proj, config, pedantic=pedantic)
    header.music = proj.constants.constantize(rom.u16(offset + 0x10), "songs", pedantic=pedantic)
    header.id = rom.u16(offset + 0x12)
    header.name_bank = proj.constants.constantize(rom.u8(offset + 0x14), "map_namespaces", pedantic=pedantic)
    header.flash_type = proj.constants.constantize(rom.u8(offset + 0x15),"map_flash_types", pedantic=pedantic)
    header.weather = proj.constants.constantize(rom.u8(offset + 0x16), "map_weathers", pedantic=pedantic)
    header.type = proj.constants.constantize(rom.u8(offset + 0x17), "map_types", pedantic=pedantic)
    header.show_name = proj.constants.constantize(rom.u8(offset + 0x19), "map_show_name_types", pedantic=pedantic)
    header.field_18 = rom.u8(offset + 0x18)
    header.field_1a = rom.u8(offset + 0x1A)
    header.battle_style = proj.constants.constantize(rom.u8(offset + 0x1B), "map_battle_styles", pedantic=pedantic)
    header.symbol = symbol
    return header

def export_footer(footer: pymap.mapfooter.Mapfooter, rom: agb.agbrom.Agbrom, offset, tsp, tss, basepath, proj, config, pedantic=False):
    """ Exports a mapfooter into a mapfooter instance"""
    footer.width = rom.u32(offset)
    footer.height = rom.u32(offset + 0x4)
    footer.border_width = rom.u8(offset + 0x18)
    footer.border_height = rom.u8(offset + 0x19)
    footer.padding = rom.u16(offset + 0x1A)
    border_off = rom.pointer(offset + 0x8)
    footer.borders = [
        [rom.u16(border_off + 2 * (y * footer.border_width + x)) for x in range(footer.border_width)]
        for y in range(footer.border_height) 
    ]
    block_off = rom.pointer(offset + 0xC)
    footer.blocks = [
        [rom.u16(block_off + 2 * (y * footer.width + x))for x in range(footer.width)]
        for y in range(footer.height)
    ]
    # Export tilesets by calling the backend os command
    tileset_config = config["backend"]["tileset"]
    tileset_base = tileset_config["tileset_base"]
    if not tsp:
        tileset_num = int((rom.pointer(offset + 0x10) - tileset_base) / 24)
        cmd = tileset_config["exec"].format(rom.path, proj.path, hex(rom.pointer(offset + 0x10)), tileset_num, basepath)
        label = tileset_config["label"].format(rom.path, proj.path, hex(rom.pointer(offset + 0x10)), tileset_num, basepath)
        if os.system(cmd):
            raise Exception("Exporting of primary tileset returned exit code != 0")
        tsp = label
    
    if not tss:
        tileset_num = int((rom.pointer(offset + 0x14) - tileset_base) / 24)
        cmd = tileset_config["exec"].format(rom.path, proj.path, hex(rom.pointer(offset + 0x14)), tileset_num, basepath)
        label = tileset_config["label"].format(rom.path, proj.path, hex(rom.pointer(offset + 0x14)), tileset_num, basepath)
        if os.system(cmd):
            raise Exception("Exporting of secondary tileset returned exit code != 0")
        tss = label
 
    # Initialize tilesets as stubs to minimize exporting time
    footer.tsp = pymap.tileset.Tileset(True)
    footer.tsp.symbol = tsp
    footer.tss = pymap.tileset.Tileset(False)
    footer.tss.symbol = tss

def export_events(header: pymap.mapheader.Mapheader, rom: agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports map events into a mapheader instance """
    person_cnt = rom.u8(offset)
    warp_cnt = rom.u8(offset + 1)
    trigger_cnt = rom.u8(offset + 2)
    signpost_cnt = rom.u8(offset + 3)
    person_off = rom.pointer(offset + 4)
    warp_off = rom.pointer(offset + 8)
    trigger_off = rom.pointer(offset + 0xC)
    signpost_off = rom.pointer(offset + 0x10)
    header.persons = [
        export_person(rom, person_off + 0x18 * i, basepath, proj, config, pedantic=pedantic) for i in range(person_cnt)
        ]
    header.warps = [
        export_warp(rom, warp_off + 0x8 * i, pedantic=pedantic) for i in range(warp_cnt)
    ]
    header.triggers = [
        export_trigger(rom, trigger_off + 0x10 * i, basepath, proj, config, pedantic=pedantic) for i in range(trigger_cnt)
    ]
    header.signposts = [
        export_sign(rom, signpost_off + 0xC * i, basepath, proj, config, pedantic=pedantic) for i in range(signpost_cnt)
    ]

        


def export_person(rom: agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports a person """
    person = mapevent.Map_event_person()
    person.target_index = rom.u8(offset)
    person.picture = rom.u8(offset + 1)
    person.field_2 = rom.u8(offset + 2)
    person.field_3 = rom.u8(offset + 3)
    person.x = rom.s16(offset + 4)
    person.y = rom.s16(offset + 6)
    person.level = rom.u8(offset + 8)
    person.behaviour = proj.constants.constantize(rom.u8(offset + 9), "person_behaviours", pedantic=pedantic)
    person.behaviour_range = rom.u8(offset + 0xA)
    person.field_b = rom.u8(offset + 0xB)
    person.is_trainer = rom.u8(offset + 0xC) & 1
    person.is_trainer_padding = rom.u8(offset + 0xC) >> 1
    person.field_d = rom.u8(offset + 0xD)
    person.alert_radius = rom.u16(offset + 0xE)

    # Export the script
    script_off = rom.nullable_pointer(offset + 0x10)
    if script_off:
        ow_script_config = config["backend"]["ow_script"]
        cmd = ow_script_config["exec"].format(rom.path, proj.path, hex(script_off), basepath, "person")
        if os.system(cmd):
            raise Exception("Execution of backend script exported produced exit code != 0")
        person.script = ow_script_config["label"].format(rom.path, proj.path, hex(script_off), basepath, "person")
    else:
        person.script = "0"

    person.flag = proj.constants.constantize(rom.u16(offset + 0x14), "flags", pedantic=pedantic)
    person.field_16 = rom.u8(offset + 0x16)
    person.field_17 = rom.u8(offset + 0x17)
    return person

def export_warp(rom: agb.agbrom.Agbrom, offset, pedantic=False):
    """ Exports a warp """
    warp = mapevent.Map_event_warp()
    warp.x = rom.s16(offset)
    warp.y = rom.s16(offset + 2)
    warp.level = rom.u8(offset + 4)
    warp.target_warp = rom.u8(offset + 5)
    warp.target_map = rom.u8(offset + 6)
    warp.target_bank = rom.u8(offset + 7)
    return warp

def export_trigger(rom : agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports a trigger """
    trigger = mapevent.Map_event_trigger()
    trigger.x = rom.s16(offset)
    trigger.y = rom.s16(offset + 2)
    trigger.level = rom.u8(offset + 4)
    trigger.field_5 = rom.u8(offset+ 5)
    trigger.variable = proj.constants.constantize(rom.u16(offset + 6), "vars", pedantic=pedantic)
    trigger.value = rom.u16(offset + 8)
    trigger.field_a = rom.u8(offset + 0xA)
    trigger.field_b = rom.u8(offset + 0xB)

    # Export the script
    script_off = rom.nullable_pointer(offset + 0xC)
    if script_off:
        ow_script_config = config["backend"]["ow_script"]
        cmd = ow_script_config["exec"].format(rom.path, proj.path, hex(script_off), basepath, "trigger")
        if os.system(cmd):
            raise Exception("Execution of backend script exported produced exit code != 0")
        trigger.script = ow_script_config["label"].format(rom.path, proj.path, hex(script_off), basepath, "trigger")
    else:
        trigger.script = "0"

    return trigger

def export_sign(rom: agb.agbrom.Agbrom, offset, basepath, proj, config, pedantic=False):
    """ Exports a sign """
    sign = mapevent.Map_event_sign()
    sign.x = rom.s16(offset)
    sign.y = rom.s16(offset + 2)
    sign.level = rom.u8(offset + 4)
    sign.sign_type = rom.u8(offset + 5)
    sign.field_6 = rom.u8(offset + 6)
    sign.field_7 = rom.u8(offset + 7)
    sign._struct_setup()
    if sign.structure == mapevent.SIGN_STRUCTURE_SCRIPT:
        # Export the script
        script_offset = rom.nullable_pointer(offset + 0x8)
        if script_offset:
            ow_script_config = config["backend"]["ow_script"]
            cmd = ow_script_config["exec"].format(rom.path, proj.path, hex(script_offset), basepath, "sign")
            if os.system(cmd):
                raise Exception("Execution of backend script exported produced exit code != 0")
            sign.script = ow_script_config["label"].format(rom.path, proj.path, hex(script_offset), basepath, "sign")
        else:
            sign.script = "0"
    else:
        sign.item_id = proj.constants.constantize(rom.u16(offset + 8), "items", pedantic=pedantic)
        sign.hidden = rom.u8(offset + 0xA)
        sign.count = rom.u8(offset + 0xB)
    return sign


if __name__ == "__main__":
    main(sys.argv[1:])