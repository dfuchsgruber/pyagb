#!/usr/bin/env python

""" This module is reponsible for building all map related files 
    directly into a rom file. (Which is not recommended at all
    since you might have extern references especially for scripts
    but some people might prefer to do this. All this module
    basically does is replace a MAKEFILE and calling repspective
    tools."""

import os, sys, getopt
from pymap import project
import tempfile
import subprocess


cwd, _ = os.path.split(__file__)
LDSCRIPT = os.path.join(cwd, "pymapbuild.ld")
LDCHUNKSIZE = 64

def main(args):

    # Setup argparser
    try:
        opts, args = getopt.getopt(args, "hi:t:o", ["help"])
    except getopt.GetoptError:
        sys.exit(2)

    proj = None
    base = None
    target = None
    offset = None

    # Parse opts
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("""Usage pybuild.py [opts] project
Opts:
    -b {base}      : Path of the base rom. Default = .config of project.
    -t {target}     : Path of the output rom. Default = .config of project.
    -o {offset}     : The location of the map data. Default = .config of project.
            """)
        elif opt in ("-i"):
            input = arg
        elif opt in ("-t"):
            target = arg
        elif opt in ("-o"):
            offset = int(arg, 0)
        else:
            raise Exception("Unkown option {0}!".format(opt))

    # Load project and configuration
    proj = project.Project.load_project(args[0])
    with open(args[0] + ".config", "r") as f:
        config = eval(f.read())["pybuild"]
    
    if not base: base = config["base"]
    if not target: target = config["target"]
    if not offset: offset = config["offset"]

    cmd_config = config["commands"]

    build(proj, base, target, offset, cmd_config["as"], cmd_config["ld"], cmd_config["grit"], cmd_config["ars"],
    cmd_config["pymap2s"], cmd_config["pyset2s"], cmd_config["pyproj2s"], config["mapbanks_ptr"], config["mapfooters_ptr"])



def build(proj, base, target, offset, cmd_as, cmd_ld, cmd_grit, cmd_ars, cmd_pymap2s, cmd_pyset2s, cmd_pyproj2s, mapbanks_ptr, mapfooters_ptr):
    """ Builds the project and referenced files. """
    
    print("Building project {0}...".format(proj.path))
    print("Creating assemblies...")
    assemblies = build_assemblies(proj, base, target, offset, cmd_as, cmd_ld, cmd_grit, cmd_ars, cmd_pymap2s, cmd_pyset2s, cmd_pyproj2s)

    # Find lscr assemblies
    print("Find levelscript assemblies...")
    lscr_assemblies = find_lscrs(os.path.dirname(proj.path))
    print("Found {0} levelscript assemblies.".format(len(lscr_assemblies)))
    assemblies += lscr_assemblies

    # Create obj files
    print("Compiling assemblies...")
    objs = build_objs(assemblies, cmd_as)

    # Link obj files
    # As the input list can not get arbitrarily long we link always in pairs of LDCHUNKSIZE files
    for i in range(0, len(objs), LDCHUNKSIZE):
        print("Chunk {0}...".format(int(i / LDCHUNKSIZE)))
        if i == 0:
            # Initial linking
            linked = link_objs(objs[i : i + LDCHUNKSIZE], offset, cmd_ld)
        else:
            # Link together with previous results
            linked = link_objs(objs[i : i + LDCHUNKSIZE] + [linked], offset, cmd_ld)
    
    # Copy linked
    print("Copying linked objects {0} -> linked.o...".format(linked.name))
    os.system("cp {0} linked.o".format(linked.name))
    with open("linked.o", "w+") as linked:
        pass

    # Create patch file
    patchfile = patch(base, target, offset, linked, cmd_ars, mapbanks_ptr, mapfooters_ptr)

    # Create makefile
    make(base, target, offset, linked, patchfile, cmd_ars)


def build_assemblies(proj, base, target, offset, cmd_as, cmd_ld, cmd_grit, cmd_ars, cmd_pymap2s, cmd_pyset2s, cmd_pyproj2s):
    """ Builds the project's assemblies (tmpfiles)"""

    assemblies = []
    # First build the map header files (pmh)
    for bank in proj.banks:
        for mapid in proj.banks[bank]:
            _, path, _, _ = proj.banks[bank][mapid]
            path = proj.realpath(path)
            # Create a new temporary file
            asfile = tempfile.NamedTemporaryFile(dir=".", delete=False, suffix=".s")
            print("Compiling {0} -> {1}...".format(path, asfile.name))
            cmd = cmd_pymap2s["command"]
            flags = cmd_pymap2s["flags"]
            os.system("{0} -o {1} {2} {3} {4}".format(cmd,
            asfile.name, flags, path, proj.path))
            assemblies.append(asfile)
    
    # Build the tileset files (pts)
    for symbol in proj.tilesets:
        path = proj.tilesets[symbol]
        path = proj.realpath(path)
        # Create a new temporary file
        asfile = tempfile.NamedTemporaryFile(dir=".", delete=True, suffix=".s")
        print("Compiling {0} -> {1}...".format(path, asfile.name))
        cmd = cmd_pyset2s["command"]
        flags = cmd_pymap2s["flags"]
        os.system("{0} -o {1} {2} {3}".format(cmd,
        asfile.name, flags, path))
        assemblies.append(asfile)
    
    # Build the project
    # Create a new temporary file
    asfile = tempfile.NamedTemporaryFile(dir=".", delete=True, suffix=".s")
    print("Compiling {0} -> {1}...".format(proj.path, asfile.name))
    cmd = cmd_pyproj2s["command"]
    flags = cmd_pyproj2s["flags"]
    os.system("{0} -b {1} -f {2} -o {3} {4} {5}".format(cmd,
    "mapbanks", "mapfooters", asfile.name, flags, proj.path))
    assemblies.append(asfile)

    # Build the pngs
    for symbol in proj.images:
        path = proj.images[symbol]
        path = proj.realpath(path)
        # Create a new temporary file
        asfile = tempfile.NamedTemporaryFile(dir=".", delete=True, suffix=".s")
        print("Compiling {0} -> {1}...".format(path, asfile.name))
        cmd = cmd_grit["command"]
        flags = cmd_grit["flags"]
        if not symbol.endswith("Tiles"):
            raise Exception("Grit does not support gfx symbols without a ...Tiles suffix. Fix symbol of gfx {0} by adding 'Tiles' as suffix!".format(symbol))
        os.system("{0} {1} -o {2} {3} -s {4}".format(cmd,
        path, asfile.name, flags, symbol[:-5]))
        assemblies.append(asfile)

    return assemblies

def find_lscrs(dir="."):
    """ Finds all lscr.asm files in dir and all subdirectories
    recursively. """
    lscr_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file == "lscr.asm":
                with open(os.path.join(root, file), "r+") as f:
                    lscr_files.append(f)
    return lscr_files


def build_objs(assemblies, cmd_as):
    """ Compiles the assemblies. """
    objs = []
    for assembly in assemblies:
        ofile = tempfile.NamedTemporaryFile(dir=".", delete=True, suffix=".o")
        print("Assembling {0} -> {1}...".format(assembly.name, ofile.name))
        cmd = cmd_as["command"]
        flags = cmd_as["flags"]
        os.system("{0} {1} -o {2} {3}".format(cmd, flags, ofile.name, assembly.name))
        objs.append(ofile)
    
    return objs

def link_objs(objs, offset, cmd_ld):
    """ Links the obj files. """
    
    # First create a ld script
    with open(LDSCRIPT) as f:
        ldscript = f.read().replace("LOCATION", hex(offset + 0x08000000))

    ldscript_file = tempfile.NamedTemporaryFile(mode="w+", dir=".", delete=True, suffix=".ld")
    ldscript_file.write(ldscript)

    # Create a second ld script for all input files except for the first one
    ldin_file = tempfile.NamedTemporaryFile(mode="w+", dir=".", delete=True, suffix=".ld")
    inputs = "INPUT({0})".format(" ".join([obj.name for obj in objs]))
    ldin_file.write(inputs)

    linked_file = tempfile.NamedTemporaryFile(dir=".", delete=True, suffix=".o")

    # Do linking
    print("Linking all objs -> {0}".format(linked_file.name))
    cmd = cmd_ld["command"]
    flags = cmd_ld["flags"]


    """    
    os.system("echo {0} {1} -T {2} -T {3} --relocatable -o {4}".format(cmd, flags, ldscript_file.name,
    ldin_file.name, linked_file.name))
    os.system("{0} {1} -T {2} -T {3} --relocatable -o {4}".format(cmd, flags, ldscript_file.name,
    ldin_file.name, linked_file.name))
    """

    #os.system("echo {0} {1} -T {2} -o {3} {4}".format(cmd, flags, ldscript_file.name, linked_file.name, " ".join([obj.name for obj in objs])))
    os.system("{0} {1} -T {2} -o {3} {4}".format(cmd, flags, ldscript_file.name, linked_file.name, " ".join([obj.name for obj in objs])))
    
    return linked_file

def patch(base, target, offset, linked, cmd_ars, mapbanks_ptr, mapfooters_ptr):
    """ Patches the file into the rom """
    print(linked.name)

    print("Patching {0} to location {1} in {2} -> {3}...".format(linked.name,
    hex(offset), base, target))
    patch = """
    .gba
    .thumb
    .open "{0}", "{1}", 0x08000000
    .org {2}
    .importobj "{3}"

    .org {4}
        .word mapbanks

    .org {5}
        .word mapfooters

    .close
    """.format(base, target, hex(offset + 0x08000000), linked.name,
    hex(mapbanks_ptr + 0x08000000), hex(mapfooters_ptr + 0x08000000))
    
    with open("patch.asm", "w+") as patchfile:
        patchfile.write(patch)
    return patchfile
    

def make(base, target, offset, linked, patchfile, cmd_ars):
    """ Creates a makefile and executes the make cmd """

    cmd = cmd_ars["command"]
    flags = cmd_ars["flags"]

    print("Creating makefile -> makefile...")
    makefile_content = """ARS=@{0}
ARSFLAGS={1}
all:
\t$(ARS) $(ARSFLAGS) {2}
clean:
\trm -f {2}
\trm -f {3}
\trm -f makefile

    """.format(cmd, flags, patchfile.name, linked.name)
    with open("makefile", "w+") as makefile:
        makefile.write(makefile_content)

    print("Created makefile:\nUse 'make {flags} all' to build!\nUse 'make {flags} clean' to remove temporary files!")


if __name__ == "__main__":
    main(sys.argv[1:])