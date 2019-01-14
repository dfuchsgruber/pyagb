from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from agb.types import *
from warnings import warn
from functools import partial

class PropertyTree(QTreeWidget):

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.setHeaderLabels(['Member', 'Value'])