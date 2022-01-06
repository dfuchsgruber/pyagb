#!/usr/bin/env python3

from pymap.gui import gui
import sys
import os

if __name__ == "__main__": 
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1' # Support for High DPI
    gui.main()