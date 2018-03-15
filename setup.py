#!/usr/bin/python3

from setuptools import setup

setup(name='pymap', 
    version='1.0', 
    description='Python3 interface for Gameboy Advance ROMs (agb)',
    author='Wodka',
    packages = ['pymap', 'agb'],
    scripts = ['bin/pymapgui.py', 'bin/pymap2s.py',
    'bin/pymapex.py', 'bin/pyproj2s.py', 
    'bin/pybuild.py', 'bin/pyset2s.py',
    'bin/pysetex.py', 'bin/pymapbatchex.py',
    'bin/pyconstex.py'],
    install_requires = ['numpy', 'Pillow', 'PyPng'],
    include_package_data=True
)
