#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(name='pyagb', 
    version='1.1', 
    description='Python3 interface for Gameboy Advance ROMs (agb)',
    author='Wodka',
    packages = find_packages(),
    scripts = [
        'script/pymapgui.py', 'script/pymap2s.py', 
        'script/pymapconstex.py', 'script/pypreproc.py', 
        'script/bin2s.py'
        ],
    install_requires = ['numpy', 'Pillow', 'PyPng', 'pyqt5', 'pyqtgraph', 'appdirs', 'scikit-image', 'deepdiff', 'scipy'],
    include_package_data=True
)
