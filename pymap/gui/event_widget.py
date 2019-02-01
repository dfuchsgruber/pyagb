from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
from PIL.ImageQt import ImageQt
import numpy as np
import properties, history, map_widget, resource_tree
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from copy import deepcopy
import os
from collections import defaultdict
from functools import partial

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
        self.tab_widget.currentChanged.connect(self.map_scene.update_selection)
        self.tabs = {}
        layout.addWidget(self.tab_widget, 1, 6, 5, 1)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(6, 1)

    def reload_project(self, *args):
        """ Called when members of the project structure are refactored, removed or inserted. Updates relevant widgets. """
        self.load_project() # There is never a discrepancy between the data model and display, so reloading the footer does not hurt the user changes

    def load_project(self):
        """ Load a new project. """
        self.load_header()
        # Create a tab for each event type
        self.tab_widget.clear()
        self.tabs = {}
        for event_type in self.main_gui.project.config['pymap']['header']['events']:
            self.tabs[event_type['datatype']] = EventTab(self, event_type)
            self.tab_widget.addTab(self.tabs[event_type['datatype']], event_type['name'])

        # Load backend for event_to_image
        project = self.main_gui.project
        backend = project.config['pymap']['display']['event_to_image_backend']
        if backend is not None:
            os.chdir(os.path.dirname(project.path))
            with open(backend) as f:
                namespace = {}
                exec(f.read(), namespace)
                self.event_to_image = namespace['event_to_image']
        else:
            self.event_to_image = lambda event, event_type, project: None # No mapping

    def load_header(self):
        """ Opens a new header. """
        self.map_scene.clear()
        self.load_map()
        self.load_events()
        if self.main_gui.project is None: return

    def load_map(self):
        """ Reloads the map image by using tiles of the map widget. """
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
        """ Loads all event images. """
        for datatype in self.tabs:
            self.tabs[datatype].load_events()
        self.map_scene.update_selection()

    def update_event(self, event_type, event_idx):
        """ Updates a certain event. """
        tab = self.tabs[event_type['datatype']]
        # Recalculate the group of this event
        self.map_scene.removeItem(self.map_scene.event_groups[event_type['datatype']][event_idx])
        event = properties.get_member_by_path(self.main_gui.header, event_type['events_path'])[event_idx]
        group = tab.event_to_group(event)
        self.map_scene.addItem(group)
        self.map_scene.event_groups[event_type['datatype']][event_idx] = group
        if tab.idx_combobox.currentIndex() == event_idx:
            # Update the properties tree
            tab.event_properties.update()
            if self.tab_widget.currentWidget() is tab:
                # Update the selection
                self.map_scene.update_selection()

def to_coordinates(x, y, padded_x, padded_y):
    """ Tries to transform the text string of an event to integer coordinates. """
    x, y = str(x), str(y) # This enables arbitrary bases
    try:
        x = int(x, 0)
        y = int(y, 0)
    except ValueError:
        return -10000, -10000 # This is hacky but prevents the events from being rendered
    return 16 * (x  + padded_x), 16 * (y + padded_y)

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
        self.add_button.clicked.connect(self.append_event)
        layout.addWidget(self.add_button, 1, 2)
        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon(resource_tree.icon_paths['remove']))
        self.remove_button.clicked.connect(lambda: self.remove_event(self.idx_combobox.currentIndex()))
        layout.addWidget(self.remove_button, 1, 3)
        self.event_properties = EventProperties(self)
        layout.addWidget(self.event_properties, 2, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 0)
        layout.setColumnStretch(3, 0)
        self.idx_combobox.currentIndexChanged.connect(self.select_event)

    def select_event(self):
        """ Selects the event of the current index. """
        self.event_widget.map_scene.update_selection()
        self.event_properties.load_event()

    def load_events(self):
        """ Updates the events according to the model. """
        self.event_properties.clear()
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None:
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
        else:
            header = self.event_widget.main_gui.header
            # Load events
            events = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])
            for event in events:
                group = self.event_to_group(event)
                self.event_widget.map_scene.event_groups[self.event_type['datatype']].append(group)
                self.event_widget.map_scene.addItem(group)
            number_events = int(properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['size_path']))
            current_idx = min(number_events - 1, max(0, self.idx_combobox.currentIndex())) # If -1 is selcted, select first, but never select a no more present event
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.addItems(list(map(str, range(number_events))))
            self.idx_combobox.setCurrentIndex(current_idx)
            self.select_event() # We want select event to be triggered even if the current idx is -1 in order to clear the properties
            self.idx_combobox.blockSignals(False)

    def remove_event(self, event_idx):
        """ Removes an event. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        if event_idx < 0: return
        self.event_widget.undo_stack.push(history.RemoveEvent(self.event_widget, self.event_type, event_idx))

    def append_event(self):
        """ Appends a new event. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        self.event_widget.undo_stack.push(history.AppendEvent(self.event_widget, self.event_type))

    def event_to_group(self, event):
        """ Creates a QGraphicsItemGroup for an event. """
        padded_x, padded_y = self.event_widget.main_gui.map_widget.get_border_padding()
        x, y = to_coordinates(properties.get_member_by_path(event, self.event_type['x_path']), properties.get_member_by_path(event, self.event_type['y_path']), padded_x, padded_y)
        # Try to get an image
        image = self.event_widget.event_to_image(event, self.event_type, self.event_widget.main_gui.project)
        if image is None or not self.event_widget.main_gui.settings['event_widget.show_pictures']:
            # Use a rectangle
            group = EventGroupRectangular(self.event_widget.map_scene, self.event_type)
            group.alignWithPosition(x, y)
        else:
            group = EventGroupImage(self.event_widget.map_scene, *image)
            group.alignWithPosition(x, y)
        return group


        

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
        event = properties.get_member_by_path(self.event_tab.event_widget.main_gui.header, self.event_tab.event_type['events_path'])[self.event_tab.idx_combobox.currentIndex()]
        self.root.blockSignals(True)
        self.root.update(event)
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
                    self.event_tab.event_widget.undo_stack.push(history.ChangeEventProperty(
                        self.event_tab.event_widget, self.event_tab.event_type, self.event_tab.idx_combobox.currentIndex(), statements_redo, statements_undo))

class MapScene(QGraphicsScene):
    """ Scene for the map view. """

    def __init__(self, event_widget, parent=None):
        super().__init__(parent=parent)
        self.event_widget = event_widget
        self.event_groups = defaultdict(list) # Items for each event type
        self.last_drag = None # Store the position where a drag happend recently so there are not multiple draw events per block
        self.dragged_event = None # Tuple to store the dragged event type and index
        self.drag_started = False # Indicate if the event has at least moved one block, i.e. if a macro is currently active

    def update_selection(self):
        """ Updates the selection rectangle to match the currently selected item. """
        if self.selection_rect is not None:
            self.removeItem(self.selection_rect)
            self.selection_rect = None
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        tab = self.event_widget.tab_widget.currentWidget()
        if tab is None: return
        idx = tab.idx_combobox.currentIndex()
        events = properties.get_member_by_path(self.event_widget.main_gui.header, tab.event_type['events_path'])
        if idx not in range(len(events)): return
        padded_x, padded_y = self.event_widget.main_gui.map_widget.get_border_padding()
        x, y = to_coordinates(properties.get_member_by_path(events[idx], tab.event_type['x_path']), properties.get_member_by_path(events[idx], tab.event_type['y_path']), padded_x, padded_y)
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        self.selection_rect = self.addRect(x, y, 16, 16, pen = QPen(color, 2.0), brush = QBrush(0))

    def clear(self):
        super().clear()
        self.event_groups = defaultdict(list)
        self.selection_rect = None

    def mouseMoveEvent(self, event):
        """ Event handler for hover events on the map image. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        padded_x, padded_y = self.event_widget.main_gui.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x - padded_x in range(map_width) and y - padded_y in range(map_height):
            self.event_widget.info_label.setText(f'x : {hex(x - padded_x)}, y : {hex(y - padded_y)}')
            if self.last_drag is not None and self.last_drag != (x, y):
                # Start the dragging macro if not started already
                if not self.drag_started:
                    self.drag_started = True
                    self.event_widget.undo_stack.beginMacro('Drag event')
                # Drag the current event to this position
                event_type, event_idx = self.dragged_event
                event = properties.get_member_by_path(self.event_widget.main_gui.header, event_type['events_path'])[event_idx]
                # Assemble undo and redo instructions for changing the coordinates
                redo_statement_x, undo_statement_x = history.path_to_statement(event_type['x_path'], self.last_drag[0] - padded_x, x - padded_x)
                redo_statement_y, undo_statement_y = history.path_to_statement(event_type['y_path'], self.last_drag[1] - padded_y, y - padded_y)
                self.event_widget.undo_stack.push(history.ChangeEventProperty(
                    self.event_widget, event_type, event_idx, [redo_statement_x, redo_statement_y], [undo_statement_x, undo_statement_y]))
                self.last_drag = x, y
        else:
            self.event_widget.info_label.setText('')


    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.event_widget.main_gui.project is None or self.event_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.event_widget.main_gui.footer, self.event_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        padded_x, padded_y = self.event_widget.main_gui.map_widget.get_border_padding()
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x - padded_x in range(map_width) and y - padded_y in range(map_height) and event.button() == Qt.LeftButton:
            # Check if there is any event that can be picked up
            for event_type in self.event_widget.main_gui.project.config['pymap']['header']['events']:
                events = properties.get_member_by_path(self.event_widget.main_gui.header, event_type['events_path'])
                for event_idx, event in enumerate(events):
                    event_x, event_y = to_coordinates(
                        properties.get_member_by_path(event, event_type['x_path']), properties.get_member_by_path(event, event_type['y_path']), padded_x, padded_y)
                    if int(event_x / 16) == x and int(event_y / 16) == y:
                        # Pick this event as new selection
                        self.event_widget.tab_widget.setCurrentWidget(self.event_widget.tabs[event_type['datatype']])
                        self.event_widget.tabs[event_type['datatype']].idx_combobox.setCurrentIndex(event_idx)
                        # Drag it until the mouse button is released
                        self.last_drag = x, y
                        self.drag_started = False # Do not start a macro unless the event is dragged at least one block
                        self.dragged_event = event_type, event_idx
                        return
                        
    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.LeftButton:
            if self.drag_started:
                # End a macro only if the event has at least moved one block
                self.event_widget.undo_stack.endMacro()
            self.drag_started = False
            self.dragged_event = None
            self.last_drag = None

class EventGroupRectangular(QGraphicsItemGroup):
    """ Subclass for an rectangular event group. """

    def __init__(self, map_scene, event_type):
        super().__init__()
        color = QColor.fromRgbF(*(event_type['box_color']))
        self.rect = map_scene.addRect(0, 0, 16, 16, pen = QPen(0), brush = QBrush(color))
        self.text = map_scene.addText(event_type['name'][0])
        font = QFont('Ubuntu')
        font.setBold(True)
        font.setPixelSize(16)
        self.text.setFont(font)
        self.text.setDefaultTextColor(QColor.fromRgbF(*(event_type['text_color'])))
        self.addToGroup(self.rect)
        self.addToGroup(self.text)

    def alignWithPosition(self, x, y):
        """ Aligns the group with a certain position """
        self.rect.setPos(x, y)
        self.text.setPos(x + 8 - self.text.sceneBoundingRect().width() / 2, y + 8 - self.text.sceneBoundingRect().height() / 2)

class EventGroupImage(QGraphicsItemGroup):
    """ Subclass for an item group consiting of only a QPixmap. """
    
    def __init__(self, map_scene, image, horizontal_displacement, vertical_displacement):
        super().__init__()
        self.horizontal_displacement = horizontal_displacement
        self.vertical_displacement = vertical_displacement
        self.pixmap = map_scene.addPixmap(QPixmap.fromImage(ImageQt(image)))
        self.addToGroup(self.pixmap)

    def alignWithPosition(self, x, y):
        """ Aligns the group with a certain position """
        self.pixmap.setPos(x + self.horizontal_displacement, y + self.vertical_displacement)
    

#def event_to_picture(event, event_type, show_pictures):
    """ Transforms an event to a picture.
    
    Parameters:
    -----------
    event : dict
        The event to draw.
    event_type : str
        The event type. 
    show_pictures : bool
        Whether the show images option is set. 
    
    Returns:
    --------
    picture : QPixmap or None
        Pixmap of the event or None if the event is not be shown as picture.
    """
    # TODO:
    #return None