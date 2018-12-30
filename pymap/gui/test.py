import os
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from pyqtgraph.parametertree import Parameter, parameterTypes
from pyqtgraph.Qt import QtCore, QtGui
import pymap.project
import json
from pprint import pprint

from pymap.gui.tree import *

os.chdir('/media/d/romhacking/Violet_Sources')


app = QtGui.QApplication([])
tree = ParameterTree()
proj = pymap.project.Project('proj.pmp')

with open(proj.headers["4"]["0"][1]) as f:
    header = json.load(f)

persons = header['data']['events']['persons']


#person = proj.model['event.person'](proj, ['test'], [])


#p = FixedSizeArrayTypeParameter('Testson', proj, 'test.fixed', proj.model['test.fixed'](proj, [], []), [], None)

"""
class MyGroup(parameterTypes.GroupParameter):

    def __init__(self, *args):
        super().__init__(name='GroupA')
        self.addChild(parameterTypes.ListParameter(name='idx', values=[0, 1, 2, 3], value=0))

        self.values = [
            Parameter.create(name=str(i), type='group', children=[{
                'name' : 'member', 'type' : 'str'
            }]) for i in range(4)
        ]
        self.update_value()
        

    def treeStateChanged(self, param, changes):
        super().treeStateChanged(param, changes)
        if param is self.child('idx'):
            # Show the element at this index
            self.update_value()
        
    def update_value(self):
        for child in self.values:
            if child.parent() is not None:
                child.remove()
        idx = int(self.child('idx').value())
        self.addChild(self.values[idx])
"""
with open('/media/d/romhacking/Violet_Sources/src/map/banks/3/1/map_3_1.pms') as f:
    ms = json.load(f)

dt = ms['type']
data = ms['data']

p = type_to_parameter(proj, dt)(ms['label'], proj, dt,  data, [], None)
tree.addParameters(p)
win = QtGui.QWidget()
layout = QtGui.QGridLayout()
win.setLayout(layout)
layout.addWidget(tree)
win.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()