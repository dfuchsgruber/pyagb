#!/usr/bin/python3

from setuptools import setup

setup(name='pyagb', 
    version='1.0', 
    description='Python3 interface for Gameboy Advance ROMs (agb)',
    author='Wodka',
    packages = ['pymap', 'agb'],
    scripts = ['bin/pymap.py', 'bin/pymap2s.py',
    'bin/pymapex.py', 'bin/pyproj2s.py', 
    'bin/pyset.py', 'bin/pyset2s.py',
    'bin/pysetex.py'],
    install_requires = ['numpy', 'Pillow', 'PyPng']
)
