#!/usr/bin/env python
from pymap import project as project
import sys
import time
import getopt
import os.path


def shell(args):
    """ Transforms a tileset file (pts) into an assembly file """
    try:
        opts, args = getopt.getopt(args, "ho:b:f:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    outfile = None
    banksym = None
    footersym = None
    for opt, arg in opts:
        if opt in ("-h", "--help"): 
            print("Usage: python pyproj2s.py -b {symbol of mapbank table} -f {symbol of footer table} -o {output assembly} projectfile")
            return
        elif opt == "-o": outfile = os.path.abspath(arg)
        elif opt == "-f": footersym = arg
        elif opt == "-b": banksym = arg


    projfile = args[0]
    if not outfile: raise Exception("No output file specified")
    if not banksym: raise Exception("No mapbank table symbol defined")
    if not footersym: raise Exception("No footer table symbol defined")
    proj = project.Project.load_project(projfile)
    fd = open(outfile, "w+")
    fd.write(export(proj, banksym, footersym, outfile))
    fd.close()

def export(proj: project.Project, banksym, footersym, outfile):
    """ Exports the two tables """
    s = "@*** Auto generated mapbank and footer tables assembly of at " + str(time.time()) + "***\n\n"
    s += ".global " + banksym + "\n.global " + footersym + "\n\n"
    
    footers = {}

    #Bank table
    s+= ".align 4\n" + banksym + ":\n"
    for bank in range(max(proj.banks) + 1):
        if bank in proj.banks:
            s += "\t .word mapbank_" + str(bank) + "\n"
        else:
            s += "\t .word 0\n"
    s += "\n\n"

    for bank in proj.banks:
        s += ".align 4\n.global mapbank_" + str(bank) + "\nmapbank_" + str(bank) + ":\n"
        for mapid in range(max(proj.banks[bank]) + 1):
            if mapid in proj.banks[bank]:
                symbol, path, namespace, footer_id = proj.banks[bank][mapid]
                if footer_id in footers: print("Warning: Collision for footer_id {0}. Following symbols collide {1} {2}.",format(str(footer_id), str(footers[footer_id]), symbol))
                footers[footer_id] = symbol
                s += "\t.word " + symbol + "\n"
            else:
                s += "\t.word 0\n"
        s += "\n\n"
    
    s += ".align 4\n" + footersym + ":\n\n"
    for footer_id in range(1, max(footers) + 1):
        if footer_id in footers:
            s += "\t.word " + footers[footer_id] + "_footer\n"
        else:
            s += "\t.word 0\n"
    s += "\n\n"
    return s
            

if __name__ == "__main__":
    shell(sys.argv[1:])