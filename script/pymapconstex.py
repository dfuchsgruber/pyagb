#!/usr/bin/env python3

""" This module is able to export the constants of a project """

from pymap import constants, project
import os, sys, getopt


def main(args):
    """ Exports the constants of a project as header / macro files """

    try:
        opts, args = getopt.getopt(args, "h:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)
    
    # Parse opts
    for opt, _ in opts:
        if opt in ("-h", "--help"): 
            print("Usage: pymapconstex.py project")
            exit(0)
    
    # Parse args
    proj = project.Project.load_project(args[0])
    
    # Load config
    with open(args[0] + ".config", "r") as f:
        config = eval(f.read())["macro"]
    
    # Exports headers / macros for each type
    for type in config:
        for label in config[type]["include"]:
            proj.constants.export_macro(label, type)
    

if __name__ == "__main__":
    main(sys.argv[1:])
