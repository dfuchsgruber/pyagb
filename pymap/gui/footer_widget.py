from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import properties, history, undo_filter
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from copy import deepcopy

class FooterWidget(ParameterTree):
    """ Class for meta-properties of the map footer (more or less if anyone desires to use those.)"""

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.root = None
        self.setEventFilter(undo_filter.UndoFilter())
        self.history = history.StateHistory(lambda: deepcopy(self.main_gui.footer))
        self.setHeaderLabels(['Property', 'Value'])
        layout = QVBoxLayout()
        self.load_project()

    def load_project(self, *args):
        """ Update project related widgets. """
        self.load_footer()

    def load_footer(self):
        """ Loads a new footer. """
        self.clear()
        if self.main_gui.project is not None and self.main_gui.header is not None and self.main_gui.footer is not None:
            footer_datatype = self.main_gui.project.config['pymap']['footer']['datatype']
            self.root = properties.type_to_parameter(self.main_gui.project, footer_datatype)(self.main_gui.footer_label, self.main_gui.project, footer_datatype, self.main_gui.footer, [], None)
            self.addParameters(self.root)
            self.root.sigTreeStateChanged.connect(self.tree_changed)
        else:
            self.root = None
        self.history.reset()

    def tree_changed(self, changes):
        diffs = DeepDiff(self.main_gui.footer, self.root.model_value())
        root = self.main_gui.footer
        for path in diffs['type_changes']:
            value = diffs['type_changes'][path]['new_value']
            exec(f'{path} = \'{value}\'')
            #print(f'{path} = {value}')
