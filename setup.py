#!/usr/bin/env python3

from setuptools import setup

setup(name='pyagb', 
    version='1.0', 
    description='Python3 interface for Gameboy Advance ROMs (agb)',
    author='Wodka',
    packages = ['pymap', 'agb', 'pokestring', 'pokescript'],
    scripts = ['script/pymapgui.py', 'script/pymap2s.py',
    'script/pymapex.py', 'script/pyproj2s.py', 
    'script/pymapbuild.py', 'script/pyset2s.py',
    'script/pysetex.py', 'script/pymapbatchex.py',
    'script/pymapconstex.py', 'script/pyowscriptex.py',
    'script/pypreproc.py', 'script/bin2s.py'],
    install_requires = ['numpy', 'Pillow', 'PyPng'],
    include_package_data=True
)
