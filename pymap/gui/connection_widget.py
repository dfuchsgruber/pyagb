from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtOpenGL import *
import numpy as np
from PIL.ImageQt import ImageQt
import map_widget, properties, render, blocks, resource_tree, history
import pyqtgraph.parametertree.ParameterTree as ParameterTree
from deepdiff import DeepDiff
from itertools import product

class ConnectionWidget(QWidget):
    """ Class to model connections. """

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

        self.connection_widget = QWidget()
        layout.addWidget(self.connection_widget, 1, 6, 5, 1)
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(6, 1)

        connection_layout = QGridLayout()
        self.connection_widget.setLayout(connection_layout)
        self.mirror_offset = QCheckBox('Mirror Offset to Adjacent Map')
        self.mirror_offset.setChecked(self.main_gui.settings['connections.mirror_offset'])
        self.mirror_offset.stateChanged.connect(self.mirror_offset_changed)
        connection_layout.addWidget(self.mirror_offset, 1, 1, 1, 3)
        self.idx_combobox = QComboBox()
        connection_layout.addWidget(self.idx_combobox, 2, 1)
        self.add_button = QPushButton()
        self.add_button.setIcon(QIcon(resource_tree.icon_paths['plus']))
        #self.add_button.clicked.connect(self.append_event)
        connection_layout.addWidget(self.add_button, 2, 2)
        self.remove_button = QPushButton()
        self.remove_button.setIcon(QIcon(resource_tree.icon_paths['remove']))
        #self.remove_button.clicked.connect(lambda: self.remove_event(self.idx_combobox.currentIndex()))
        connection_layout.addWidget(self.remove_button, 2, 3)
        self.connection_properties = ConnectionProperties(self)
        connection_layout.addWidget(self.connection_properties, 3, 1, 1, 3)
        connection_layout.setColumnStretch(1, 1)
        connection_layout.setColumnStretch(2, 0)
        connection_layout.setColumnStretch(3, 0)
        self.idx_combobox.currentIndexChanged.connect(self.select_connection)

    def mirror_offset_changed(self):
        """ Event handler for when the mirror offset checkbox is toggled. """
        self.main_gui.settings['connections.mirror_offset'] = self.mirror_offset.isChecked()

    def load_project(self):
        """ Loads a new project. """
        self.load_header()

    def load_header(self):
        """ Loads graphics for the current header. """
        self.map_scene.clear()
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None:
            self.idx_combobox.blockSignals(True)
            self.idx_combobox.clear()
            self.idx_combobox.blockSignals(False)
            return
        self.base_blocks = blocks.compute_blocks(self.main_gui.footer, self.main_gui.project)
        # Load connections
        self.connections = blocks.unpack_connections(
            properties.get_member_by_path(self.main_gui.header, self.main_gui.project.config['pymap']['header']['connections']['connections_path']), self.main_gui.project)

        # Load the current blocks
        self.blocks = self.compute_blocks()
        self.block_pixmaps = np.empty_like(self.blocks[:, :, 0], dtype=object)
        for (y, x), block_idx in np.ndenumerate(self.blocks[:, :, 0]):
            # Draw the blocks
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[block_idx]))
            item = QGraphicsPixmapItem(pixmap)
            item.setAcceptHoverEvents(True)
            self.map_scene.addItem(item)
            item.setPos(16 * x, 16 * y)
            self.block_pixmaps[y, x] = item

        padded_width, padded_height = self.main_gui.project.config['pymap']['display']['border_padding']
        map_width = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_height_path'])

        # Draw rectangles for the borders
        border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['border_color']))
        self.map_scene.addRect(0, 0, 16 * (map_width + 2 * padded_width), 16 * padded_height, pen=QPen(0), brush=QBrush(border_color))
        self.map_scene.addRect(0, 16 * padded_height, 16 * padded_width, 16 * map_height, pen=QPen(0), brush=QBrush(border_color))
        self.map_scene.addRect(16 * (padded_width + map_width), 16 * padded_height, 16 * padded_width, 16 * map_height, pen=QPen(0), brush=QBrush(border_color))
        self.map_scene.addRect(0, 16 * (padded_height + map_height), 16 * (map_width + 2 * padded_width), 16 * padded_height, pen=QPen(0), brush=QBrush(border_color))

        # Setup zero size rectangles for all possible connections
        connection_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_color']))
        connection_border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_border_color']))
        self.connection_rects = {direction : self.map_scene.addRect(0, 0, 0, 0, pen=QPen(connection_border_color), brush=QBrush(connection_color)) for direction in ('north', 'south', 'east', 'west')}
        self.update_border_rectangles()

        self.map_scene.setSceneRect(0, 0, 16 * (map_width + 2 * padded_width), 16 * (map_height + 2 * padded_height))

        current_idx = min(len(self.connections) - 1, max(0, self.idx_combobox.currentIndex())) # If -1 is selcted, select first, but never select a no more present event
        self.idx_combobox.blockSignals(True)
        self.idx_combobox.clear()
        self.idx_combobox.addItems(list(map(str, range(len(self.connections)))))
        self.idx_combobox.setCurrentIndex(current_idx)
        self.select_connection() # We want select connection to be triggered even if the current idx is -1 in order to clear the properties
        self.idx_combobox.blockSignals(False)
    
    def select_connection(self):
        """ Selects the event of the current index. """
        self.connection_properties.load_connection()
        self.update_border_rectangles()
    
    def compute_blocks(self):
        """ Comptues the current block map with the current connections. """
        map_blocks = self.base_blocks.copy()
        for connection in blocks.filter_visible_connections(self.connections):
            blocks.insert_connection(map_blocks, connection, self.main_gui.footer, self.main_gui.project)
        return map_blocks

    def update_connection(self, connection_idx, mirror_offset):
        """ Updates a certain connection. """
        if self.idx_combobox.currentIndex() == connection_idx:
            self.connection_properties.update()
        packed = properties.get_member_by_path(self.main_gui.header, self.main_gui.project.config['pymap']['header']['connections']['connections_path'])[connection_idx]
        # Update the unpacked version
        self.connections[connection_idx] = blocks.unpack_connection(packed, self.main_gui.project)
        self.update_blocks()
        self.update_border_rectangles()
        # Mirror offset changes to the adjacent map
        if mirror_offset and self.connections[connection_idx] is not None:
            connection_type, offset, bank, map_idx, connection_blocks = self.connections[connection_idx]
            # Load the adjacent header
            header, _, _ = self.main_gui.project.load_header(bank, map_idx)
            if header is not None:
                # Find the correlating connection
                adjacent_packed = properties.get_member_by_path(header, self.main_gui.project.config['pymap']['header']['connections']['connections_path'])
                adjacent_connections = blocks.unpack_connections(adjacent_packed, self.main_gui.project, default_blocks=np.empty((0, 0, 2), dtype=int))
                for idx, adjacent_connection in enumerate(adjacent_connections):
                    # Check if the adjacent bank and map idx match the current map
                    if adjacent_connection is None: continue
                    adjacent_connection_type, adjacent_offset, adjacent_bank, adjacent_map_idx, _ = adjacent_connection
                    # Bring bank and map_idx in their canonical forms
                    try: adjacent_map_idx = str(int(str(adjacent_map_idx), 0))
                    except ValueError: pass
                    try: adjacent_bank = str(int(str(adjacent_bank), 0))
                    except ValueError: pass
                    if opposite_directions[connection_type] == adjacent_connection_type and adjacent_bank == self.main_gui.header_bank and adjacent_map_idx == self.main_gui.header_map_idx:
                        # Match, mirror the change
                        properties.set_member_by_path(adjacent_packed[idx], str(-offset), self.main_gui.project.config['pymap']['header']['connections']['connection_offset_path'])
                        print(-offset)
                        self.main_gui.project.save_header(header, bank, map_idx)

    def update_blocks(self):    
        """ Visually updates the blocks and connections. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None: return
        new_blocks = self.compute_blocks()
        # Check which blocks have changed and only render those
        for (y, x) in zip(*np.where(new_blocks[:, :, 0] != self.blocks[:, :, 0])):
            pixmap = QPixmap.fromImage(ImageQt(self.main_gui.blocks[new_blocks[y, x, 0]]))
            self.block_pixmaps[y, x].setPixmap(pixmap)
        self.blocks = new_blocks

    def update_border_rectangles(self):
        """ Updates the border rectangles. """
        # First hide all rectangles
        for direction in self.connection_rects:
            self.connection_rects[direction].setRect(0, 0, 0, 0)
        # Show border rectangles
        padded_width, padded_height = self.main_gui.project.config['pymap']['display']['border_padding']
        map_width = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.main_gui.footer, self.main_gui.project.config['pymap']['footer']['map_height_path'])
        connection_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_color']))
        connection_active_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_active_color']))
        connection_border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_border_color']))
        connection_active_border_color = QColor.fromRgbF(*(self.main_gui.project.config['pymap']['display']['connection_active_border_color']))
        for idx, connection in enumerate(blocks.filter_visible_connections(self.connections, keep_invisble=True)):
            if connection is None: continue
            direction, offset, bank, map_idx, connection_blocks = connection
            connection_width, connection_height = connection_blocks.shape[1], connection_blocks.shape[0]
            if direction == 'north':
                rect = 16 * (padded_width + offset), 16 * (padded_height - connection_height), 16 * connection_width, 16 * connection_height
            elif direction == 'south':
                rect = 16 * (padded_width + offset), 16 * (padded_height + map_height), 16 * connection_width, 16 * connection_height
            elif direction == 'east':
                rect = 16 * (padded_width + map_width), 16 * (padded_height + offset), 16 * connection_width, 16 * connection_height
            elif direction == 'west':
                rect = 16 * (padded_width - connection_width), 16 * (padded_height + offset), 16 * connection_width, 16 * connection_height
            self.connection_rects[direction].setRect(*fix_rect(*rect, 16 * (map_width + 2 * padded_width), 16 * (map_height + 2 * padded_height)))
            if idx == self.idx_combobox.currentIndex():
                self.connection_rects[direction].setPen(QPen(connection_active_border_color))
                self.connection_rects[direction].setBrush(QBrush(connection_active_color))
            else:
                self.connection_rects[direction].setPen(QPen(connection_border_color))
                self.connection_rects[direction].setBrush(QBrush(connection_color))

        self.map_scene.setSceneRect(0, 0, 16 * (map_width + 2 * padded_width), 16 * (map_height + 2 * padded_height))

def fix_rect(x, y, width, height, max_width, max_height):
    """ Fixes the position of a rectangle to fit into the graphics scene. """
    # Fix negative bounds
    x, width = max(0, x), width + min(0, x)
    y, height = max(0, y), height + min(0, y)
    # Fix positive bounds
    if x + width > max_width:
        width = max_width - x
    if y + height > max_height:
        height = max_height - y
    # If width or height became negative, do not show the rect
    width, height = max(0, width), max(0, height)
    return x, y, width, height

opposite_directions = {
    'north' : 'south',
    'south' : 'north',
    'east' : 'west',
    'west' : 'east',
}

class ConnectionProperties(ParameterTree):
    """ Tree to display event properties. """

    def __init__(self, connection_widget, parent=None):
        super().__init__(parent=parent)
        self.connection_widget = connection_widget
        self.setHeaderLabels(['Property', 'Value'])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.root = None

    def load_connection(self):
        """ Loads the currently displayed connection. """
        self.clear()
        if self.connection_widget.main_gui.project is None or self.connection_widget.main_gui.header is None or self.connection_widget.idx_combobox.currentIndex() < 0:
            self.root = None
        else:
            datatype = self.connection_widget.main_gui.project.config['pymap']['header']['connections']['datatype']
            connections = properties.get_member_by_path(self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
            connection = connections[self.connection_widget.idx_combobox.currentIndex()]
            self.root = properties.type_to_parameter(self.connection_widget.main_gui.project, datatype)(
                '.', self.connection_widget.main_gui.project, datatype, connection, 
                self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'] + [self.connection_widget.idx_combobox.currentIndex()], connections
                )
            self.addParameters(self.root, showTop=False)
            self.root.sigTreeStateChanged.connect(self.tree_changed)

    def update(self):
        """ Updates all values in the tree according to the current connection. """
        connection = properties.get_member_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path']
            )[self.connection_widget.idx_combobox.currentIndex()]
        self.root.blockSignals(True)
        self.root.update(connection)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        connections = properties.get_member_by_path(self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        root = connections[self.connection_widget.idx_combobox.currentIndex()]
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
                    self.connection_widget.undo_stack.push(history.ChangeConnectionProperty(
                        self.connection_widget, self.connection_widget.idx_combobox.currentIndex(), self.connection_widget.mirror_offset.isChecked(), statements_redo, statements_undo))


class MapScene(QGraphicsScene):
    """ Scene for the map view. """

    def __init__(self, connection_widget, parent=None):
        super().__init__(parent=parent)
        self.connection_widget = connection_widget
        self.last_drag = None # Store the position where a drag happend recently so there are not multiple draw events per block
        self.dragged_idx = -1 # Store the index of the connection that is dragged
        self.drag_started = False # Indicate if the connection has at least moved one block, i.e. if a macro is currently active
        self.drag_origin = None # Store the position where the drag originated in order to calculate an offset

    def clear(self):
        super().clear()

    def mouseMoveEvent(self, event):
        """ Event handler for hover events on the map image. """
        if self.connection_widget.main_gui.project is None or self.connection_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.connection_widget.main_gui.footer, self.connection_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.connection_widget.main_gui.footer, self.connection_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        padded_x, padded_y = self.connection_widget.main_gui.project.config['pymap']['display']['border_padding']
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x - padded_x in range(map_width) and y - padded_y in range(map_height):
            self.connection_widget.info_label.setText(f'x : {hex(x - padded_x)}, y : {hex(y - padded_y)}')
            return
        
            if self.last_drag is not None and self.last_drag != (x, y):
                # Start the dragging macro if not started already
                if not self.drag_started:
                    self.drag_started = True
                    self.connection_widget.undo_stack.beginMacro('Drag event')
                # Drag the current event to this position
                event_type, event_idx = self.dragged_event
                event = properties.get_member_by_path(self.connection_widget.main_gui.header, event_type['events_path'])[event_idx]
                # Assemble undo and redo instructions for changing the coordinates
                x_path = ''.join(map(lambda member: f'[{repr(member)}]', event_type['x_path']))
                y_path = ''.join(map(lambda member: f'[{repr(member)}]', event_type['y_path']))
                redo_statements = [f'root{x_path} = \'{x - padded_x}\'', f'root{y_path} = \'{y - padded_y}\'']
                undo_statements = [f'root{x_path} = \'{self.last_drag[0] - padded_x}\'', f'root{y_path} = \'{self.last_drag[1] - padded_y}\'']
                self.connection_widget.undo_stack.push(history.ChangeEventProperty(
                    self.connection_widget, event_type, event_idx, redo_statements, undo_statements))
                self.last_drag = x, y
        else:
            self.connection_widget.info_label.setText('')


    def mousePressEvent(self, event):
        """ Event handler for pressing the mouse. """
        if self.connection_widget.main_gui.project is None or self.connection_widget.main_gui.header is None: return
        map_width = properties.get_member_by_path(self.connection_widget.main_gui.footer, self.connection_widget.main_gui.project.config['pymap']['footer']['map_width_path'])
        map_height = properties.get_member_by_path(self.connection_widget.main_gui.footer, self.connection_widget.main_gui.project.config['pymap']['footer']['map_height_path'])
        padded_x, padded_y = self.connection_widget.main_gui.project.config['pymap']['display']['border_padding']
        pos = event.scenePos()
        x, y = int(pos.x() / 16), int(pos.y() / 16)
        if x in range(2 * padded_x + map_width) and y in range(2 * padded_y + map_height) and event.button() == Qt.LeftButton:
            # Check if there is any connection
            self.dragged_idx = -1
            for direction in self.connection_widget.connection_rects:
                rect = self.connection_widget.connection_rects[direction].rect()
                if x >= int(rect.x() / 16) and x < int((rect.x() + rect.width()) / 16) and y >= int(rect.y() / 16) and y < int((rect.y() + rect.height()) / 16):
                    # Find the index that matches direction
                    for idx, connection in enumerate(self.connection_widget.connections):
                        if connection is not None and direction == connection[0]:
                            self.dragged_idx = idx
                            break
                    if self.dragged_idx == -1:
                        raise RuntimeError(f'Inconsistent connection type {direction}. Did not find any matching connection for a rectangle.')
                    self.connection_widget.idx_combobox.setCurrentIndex(self.dragged_idx)
                    self.last_drag = x, y
                    self.drag_origin = x, y
                    self.drag_started = False
                        
    def mouseReleaseEvent(self, event):
        """ Event handler for releasing the mouse. """
        if event.button() == Qt.LeftButton:
            if self.drag_started:
                # End a macro only if the event has at least moved one block
                self.connection_widget.undo_stack.endMacro()
            self.drag_started = False
            self.dragged_idx = -1
            self.last_drag = None
            self.drag_origin = None