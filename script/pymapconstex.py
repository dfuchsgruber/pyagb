#!/usr/bin/env python3

""" This module is able to export the constants of a project """

from pymap import constants, project
import os, sys, getopt


def main(args):
    """ Exports the constants of a project as header / macro files """

    try:
        opts, args = getopt.getopt(args, "h:", ["help", "get"])
    except getopt.GetoptError:
        sys.exit(2)
    
    get = False
    # Parse opts
    for opt, _ in opts:
        if opt in ("-h", "--help"): 
            print("Usage: pymapconstex.py project\n\nUse --get to only output all files")
            exit(0)
        elif opt in ("--get"):
            get = True
    
    # Parse args
    proj = project.Project.load_project(args[0])
    
    # Load config
    with open(args[0] + ".config", "r") as f:
        config = eval(f.read())["macro"]
    
    if get:
        paths = []

    # Exports headers / macros for each type
    for type in config:
        for label in config[type]["include"]:
            if get:
                # Append the path of this label
                path = proj.constants.macro_conf[type]["path"].format(label)
                paths.append(path)
            else:
                proj.constants.export_macro(label, type)
    
    if get:
        print(" ".join(paths))


if __name__ == "__main__":
    main(sys.argv[1:])
