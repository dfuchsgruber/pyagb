#!/usr/bin/env python3

from setuptools import setup

setup(name='pymap', 
    version='1.0', 
    description='Python3 interface for Gameboy Advance ROMs (agb)',
    author='Wodka',
    packages = ['pymap', 'agb'],
    scripts = ['script/pymapgui.py', 'script/pymap2s.py',
    'script/pymapex.py', 'script/pyproj2s.py', 
    'script/pybuild.py', 'script/pyset2s.py',
    'script/pysetex.py', 'script/pymapbatchex.py',
    'script/pyconstex.py'],
    install_requires = ['numpy', 'Pillow', 'PyPng'],
    include_package_data=True
)
