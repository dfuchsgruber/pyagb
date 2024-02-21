from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
from . import properties, history
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from copy import deepcopy

class HeaderWidget(ParameterTree):
    """ Class for meta-properties of the map footer (more or less if anyone desires to use those.)"""

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.root = None
        self.undo_stack = QtGui.QUndoStack()
        self.setHeaderLabels(['Property', 'Value'])
        self.header().setSectionResizeMode(QHeaderView.Interactive)
        self.header().setStretchLastSection(True)
        layout = QVBoxLayout()
        self.load_project()

    def load_project(self, *args):
        """ Update project related widgets. """
        self.load_header()
        
    def load_header(self):
        """ Loads a new header. """
        self.clear()
        if self.main_gui.project is not None and self.main_gui.header is not None:
            header_datatype = self.main_gui.project.config['pymap']['header']['datatype']
            header_label = self.main_gui.project.headers[self.main_gui.header_bank][self.main_gui.header_map_idx][0]
            self.root = properties.type_to_parameter(self.main_gui.project, header_datatype)(header_label, self.main_gui.project, header_datatype, self.main_gui.header, [], None)
            self.addParameters(self.root)
            self.root.sigTreeStateChanged.connect(self.tree_changed)
        else:
            self.root = None

    def update(self):
        """ Updates all values in the tree according to the current footer. """
        self.root.blockSignals(True)
        self.root.update(self.main_gui.header)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        root = self.main_gui.header
        diffs = DeepDiff(root, self.root.model_value())
        statements_redo = []
        statements_undo = []
        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:
                    value_new = diffs[change][path]['new_value']
                    value_old = diffs[change][path]['old_value']
                    statements_redo.append(f'{path} = \'{value_new}\'')
                    statements_undo.append(f'{path} = \'{value_old}\'')
        self.undo_stack.push(history.ChangeHeaderProperty(self, statements_redo, statements_undo))
                    # exec(f'{path} = \'{value}\'')
                    # print(f'{path} = {value}')


