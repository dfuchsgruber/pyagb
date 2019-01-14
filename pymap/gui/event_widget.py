from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import numpy as np
import properties, history, map_widget, resource_tree
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from copy import deepcopy
import os

class EventWidget(QWidget):
    """ Class to model events. """

    def __init__(self, main_gui, parent=None):
        super().__init__(parent=parent)
        self.main_gui = main_gui
        self.undo_stack = QUndoStack()

        # Layout is similar to the map widget
        layout = QGridLayout()
        self.setLayout(layout)

        self.map_scene = MapScene(self)
        self.map_scene_view = QGraphicsView()
        self.map_scene_view.setViewport(QGLWidget())
        self.map_scene_view.setScene(self.map_scene)
        layout.addWidget(self.map_scene_view, 1, 1, 5, 5)

        self.info_label = QLabel()
        layout.addWidget(self.info_label, 6, 1, 1, 6)
        layout.setRowStretch(1, 5)
        layout.setRowStretch(6, 0)

        self.tab_widget = QTabWidget()
        self.tabs = {}
        layout.addWidget(self.tab_widget, 1, 6, 5, 1)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(6, 1)

    def load_project(self):
        """ Load a new project. """
        self.load_header()
        # Create a tab for each event type
        self.tab_widget.clear()
        self.tabs = {}
        for event_type in self.main_gui.project.config['pymap']['header']['events']:
            self.tabs[event_type['datatype']] = EventTab(self, event_type)
            self.tab_widget.addTab(self.tabs[event_type['datatype']], event_type['name'])

    def load_header(self):
        """ Opens a new header. """
        self.load_map()
        self.load_events()
        if self.main_gui.project is None: return

    def load_map(self):
        """ Reloads the map image by using tiles of the map widget. """
        self.map_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None: return
        # Load pixel maps directly from the map widget
        for (y, x), item in np.ndenumerate(self.main_gui.map_widget.map_images):
            pixmap = item.pixmap()
            item = QGraphicsPixmapItem(pixmap) # Create a new item bound to this canvas
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
        # Load rectangles directly from the map widget
        border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['border_color']))
        self.map_scene.addRect(self.main_gui.map_widget.north_border.rect(), pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.addRect(self.main_gui.map_widget.south_border.rect(), pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.addRect(self.main_gui.map_widget.east_border.rect(), pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.addRect(self.main_gui.map_widget.west_border.rect(), pen = QPen(0), brush = QBrush(border_color))
        self.map_scene.setSceneRect(0, 0, 16 * self.main_gui.map_widget.map_images.shape[1], 16 * self.main_gui.map_widget.map_images.shape[0])

    def load_events(self):
        """ Reloads all event images. """
        for datatype in self.tabs:
            self.tabs[datatype].load_events()


class EventTab(QWidget):
    """ Tab for an event type. """

    def __init__(self, event_widget, event_type, parent=None):
        super().__init__(parent=parent)
        self.event_widget = event_widget
        self.event_type = event_type

        layout = QGridLayout()
        self.setLayout(layout)
        self.idx_combobox = QComboBox()
        layout.addWidget(self.idx_combobox, 1, 1)
        self.add_button = QPushButton()
        self.add_button.setIcon(QIcon(resource_tree.icon_paths['plus']))
        layout.addWidget(self.add_button, 1, 2)
        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon(resource_tree.icon_paths['remove']))
        layout.addWidget(self.remove_button, 1, 3)
        self.event_properties = EventProperties(self)
        layout.addWidget(self.event_properties, 2, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        self.idx_combobox.currentIndexChanged.connect(self.event_properties.load_event)

    def load_events(self):
        """ Updates the events according to the model. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None:
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
        else:
            header = self.event_widget.main_gui.header
            number_events = int(properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['size_path']))
            current_idx = min(number_events - 1, max(0, self.idx_combobox.currentIndex())) # If -1 is selcted, select first, but never select a no more present event
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.addItems(list(map(str, range(number_events))))
            self.idx_combobox.setCurrentIndex(current_idx)
            self.idx_combobox.blockSignals(False)
            self.event_properties.load_event()


class EventProperties(ParameterTree):
    """ Tree to display event properties. """

    def __init__(self, event_tab, parent=None):
        super().__init__(parent=parent)
        self.event_tab = event_tab
        self.setHeaderLabels(['Property', 'Value'])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.root = None

    def load_event(self):
        """ Loads the currently displayed event. """
        self.clear()
        if self.event_tab.event_widget.main_gui.project is None or self.event_tab.event_widget.main_gui.header is None or self.event_tab.idx_combobox.currentIndex() < 0:
            self.root = None
        else:
            datatype = self.event_tab.event_type['datatype']
            events = properties.get_member_by_path(self.event_tab.event_widget.main_gui.header, self.event_tab.event_type['events_path'])
            event = events[self.event_tab.idx_combobox.currentIndex()]
            self.root = properties.type_to_parameter(self.event_tab.event_widget.main_gui.project, datatype)(
                '.', self.event_tab.event_widget.main_gui.project, datatype, event, self.event_tab.event_type['events_path'] + [self.event_tab.idx_combobox.currentIndex()], events)
            self.addParameters(self.root, showTop=False)
            self.root.sigTreeStateChanged.connect(self.tree_changed)
        

    def update(self):
        """ Updates all values in the tree according to the current event. """
        self.root.blockSignals(True)
        self.root.update(self.main_gui.footer)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        events = properties.get_member_by_path(self.event_tab.event_widget.main_gui.header, self.event_tab.event_type['events_path'])
        root = events[self.event_tab.idx_combobox.currentIndex()]
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
                    print(statements_redo)
                    print(statements_undo)


class MapScene(QGraphicsScene):
    """ Scene for the map view. """

    def __init__(self, event_widget, parent=None):
        super().__init__(parent=parent)
        self.event_widget = event_widget
        self.selection_box = None
        self.last_draw = None # Store the position where a draw happend recently so there are not multiple draw events per block

    def mouseMoveEvent(self, event):
        """ Event handler for hover events on the map image. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        border_width, border_height = self.event_widget.main_gui.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)

    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        border_width, border_height = self.event_widget.main_gui.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)

    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        pass