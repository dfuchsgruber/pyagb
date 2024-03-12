# Widget to display a map and its events

from typing import Sequence
from . import render
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import agb.image
from PIL.ImageQt import ImageQt
import pyqtgraph.parametertree.ParameterTree as ParameterTree
import numpy as np
from skimage.measure import label
from . import properties, history, blocks, smart_shape
import os
import json
from collections import OrderedDict
