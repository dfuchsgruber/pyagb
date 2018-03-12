#!/usr/bin/python3

""" This module is essential as it extends paths relative to the tools directory to be relative to the root (the makefile)"""
import os.path

ROOT = "tools/"
THIS = "../"

def rootpath(p):
    """ Extends a path relative to the tools directory to be relative to root directory"""
    return ROOT + os.path.relpath(p).replace("\\", "/")

def path(p, from_root):
    return rootpath(p) if from_root else os.path.relpath(p)