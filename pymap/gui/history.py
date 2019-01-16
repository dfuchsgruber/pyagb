import properties
from warnings import warn
from deepdiff import DeepDiff
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Resize(QUndoCommand):
    """ Resizes a set of blocks. """

    def __init__(self, main_gui, height_new, width_new, values_old):
        super().__init__()
        self.main_gui = main_gui
        self.height_old, self.width_old = values_old.shape[0], values_old.shape[1]
        self.height_new, self.width_new = height_new, width_new
        height, width = max(self.height_new, self.height_old), max(self.width_new, self.width_old)
        self.buffer = np.zeros((height, width, 2), dtype=int)
        self.buffer[ : self.height_old, : self.width_old, :] = values_old.copy()

    def _old_blocks(self):
        """ Returns a copy of the old blocks. """
        return self.buffer[ : self.height_old, : self.width_old, :].copy()
    
    def _new_blocks(self):
        """ Returns a copy of the new blocks. """
        return self.buffer[ : self.height_new, : self.width_new, :].copy()

class ResizeMap(Resize):
    """ Action for resizing the map blocks. """

    def _change_size(self, blocks):
        properties.set_member_by_path(self.main_gui.footer, blocks.shape[0], self.main_gui.project.config['pymap']['footer']['map_height_path'])
        properties.set_member_by_path(self.main_gui.footer, blocks.shape[1], self.main_gui.project.config['pymap']['footer']['map_width_path'])
        properties.set_member_by_path(self.main_gui.footer, blocks, self.main_gui.project.config['pymap']['footer']['map_blocks_path'])
        self.main_gui.map_widget.load_header()

    def redo(self):
        """ Resizes the map blocks. """
        self._change_size(self._new_blocks())

    def undo(self):
        """ Undoes resizing of the map blocks. """
        self._change_size(self._old_blocks())

class ResizeBorder(Resize):
    """ Action for resizing the map border. """

    def _change_size(self, blocks):
        properties.set_member_by_path(self.main_gui.footer, blocks.shape[0], self.main_gui.project.config['pymap']['footer']['border_height_path'])
        properties.set_member_by_path(self.main_gui.footer, blocks.shape[1], self.main_gui.project.config['pymap']['footer']['border_width_path'])
        properties.set_member_by_path(self.main_gui.footer, blocks, self.main_gui.project.config['pymap']['footer']['border_path'])
        self.main_gui.map_widget.load_header()

    def redo(self):
        """ Resizes the map blocks. """
        self._change_size(self._new_blocks())

    def undo(self):
        """ Undoes resizing of the map blocks. """
        self._change_size(self._old_blocks())

class SetBorder(QUndoCommand):
    """ Action for setting border blocks. """

    def __init__(self, main_gui, x, y, blocks_new, blocks_old):
        super().__init__()
        self.main_gui = main_gui
        self.x = x
        self.y = y
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks):
        # Helper method for setting a set of blocks
        properties.get_member_by_path(
            self.main_gui.footer, 
            self.main_gui.project.config['pymap']['footer']['border_path'])[
                self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], 0] = blocks[:, :, 0]
        self.main_gui.map_widget.load_map()
        self.main_gui.map_widget.load_border()

    def redo(self):
        """ Performs the setting of border blocks. """
        self._set_blocks(self.blocks_new)
    
    def undo(self):
        """ Undos setting of border blocks. """
        self._set_blocks(self.blocks_old)

class SetBlocks(QUndoCommand):
    """ Action for setting blocks. """

    def __init__(self, main_gui, x, y, layers, blocks_new, blocks_old):
        super().__init__()
        self.main_gui = main_gui
        self.x = x
        self.y = y
        self.layers = layers
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks):
        # Helper method for setting a set of blocks
        properties.get_member_by_path(
            self.main_gui.footer, 
            self.main_gui.project.config['pymap']['footer']['map_blocks_path'])[
                self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], self.layers] = blocks[:, :, self.layers]
        self.main_gui.map_widget.update_map(self.x, self.y, self.layers, blocks)

    def redo(self):
        """ Performs the setting of blocks. """
        self._set_blocks(self.blocks_new)
    
    def undo(self):
        """ Undos setting of blocks. """
        self._set_blocks(self.blocks_old)

class ReplaceBlocks(QUndoCommand):
    """ Action for replacing a set of blocks with another. """

    def __init__(self, main_gui, idx, layer, value_new, value_old):
        super().__init__()
        self.main_gui = main_gui
        self.idx = idx
        self.layer = layer
        self.value_new = value_new
        self.value_old = value_old

    def redo(self):
        """ Performs the flood fill. """
        map_blocks = properties.get_member_by_path(
            self.main_gui.footer, 
            self.main_gui.project.config['pymap']['footer']['map_blocks_path'])
        map_blocks[:, :, self.layer][self.idx] = self.value_new
        self.main_gui.map_widget.load_map()

    def undo(self):
        """ Performs the flood fill. """
        map_blocks = properties.get_member_by_path(
            self.main_gui.footer, 
            self.main_gui.project.config['pymap']['footer']['map_blocks_path'])
        map_blocks[:, :, self.layer][self.idx] = self.value_old
        self.main_gui.map_widget.load_map()

class AssignTileset(QUndoCommand):
    """ Class for assigning a new tileset. """
    
    def __init__(self, main_gui, primary, label_new, label_old):
        super().__init__()
        self.main_gui = main_gui
        self.label_new = label_new
        self.label_old = label_old
        self.primary = primary

    def _assign(self, label):
        """ Helper for assigning a label. """
        if self.main_gui.project is None: return
        if not label in getattr(self.main_gui.project, ('tilesets_primary' if self.primary else 'tilesets_secondary')):
            return QMessageBox.critical(self.main_gui, 'Unable to assign tileset', f'Unable to assign tileset {label} since it is no longer in the project. Maybe you deleted it?')
        if self.primary:
            self.main_gui.open_tilesets(label_primary=label)
        else:
            self.main_gui.open_tilesets(label_secondary=label)
    
    def redo(self):
        """ Performs the tileset assignment. """
        self._assign(self.label_new)

    def undo(self):
        """ Undoes the tileset assignment. """
        self._assign(self.label_old)

class ChangeFooterProperty(QUndoCommand):
    """ Change a property of the footer. """

    def __init__(self, footer_widget, statements_redo, statements_undo):
        super().__init__()
        self.footer_widget = footer_widget
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """ Executes the redo statements. """
        root = self.footer_widget.main_gui.footer
        for statement in self.statements_redo:
            exec(statement)
        self.footer_widget.update()

    def undo(self):
        """ Executes the redo statements. """
        root = self.footer_widget.main_gui.footer
        for statement in self.statements_undo:
            exec(statement)
        self.footer_widget.update()
    
class ChangeHeaderProperty(QUndoCommand):
    """ Change a property of the header. """

    def __init__(self, header_widget, statements_redo, statements_undo):
        super().__init__()
        self.header_widget = header_widget
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """ Executes the redo statements. """
        root = self.header_widget.main_gui.header
        for statement in self.statements_redo:
            exec(statement)
        self.header_widget.update()

    def undo(self):
        """ Executes the redo statements. """
        root = self.header_widget.main_gui.header
        for statement in self.statements_undo:
            exec(statement)
        self.header_widget.update()
    
class ChangeEventProperty(QUndoCommand):
    """ Change a property of any vent. """

    def __init__(self, event_widget, event_type, event_idx, statements_redo, statements_undo):
        super().__init__()
        self.event_widget = event_widget
        self.event_type = event_type
        self.event_idx = event_idx
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """ Executes the redo statements. """
        root = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])[self.event_idx]
        for statement in self.statements_redo:
            exec(statement)
        self.event_widget.update_event(self.event_type, self.event_idx)

    def undo(self):
        """ Executes the redo statements. """
        root = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])[self.event_idx]
        for statement in self.statements_undo:
            exec(statement)
        self.event_widget.update_event(self.event_type, self.event_idx)

class RemoveEvent(QUndoCommand):
    """ Remove an event. """

    def __init__(self, event_widget, event_type, event_idx):
        super().__init__()
        self.event_widget = event_widget
        self.event_type = event_type
        self.event_idx = event_idx
        self.event = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])[self.event_idx]
    
    def redo(self):
        """ Removes the event from the events. """
        events = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])
        events.pop(self.event_idx)
        properties.set_member_by_path(self.event_widget.main_gui.header, len(events), self.event_type['size_path'])
        self.event_widget.load_header()

    def undo(self):
        """ Reinserts the event. """
        events = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])
        events.insert(self.event_idx, self.event)
        properties.set_member_by_path(self.event_widget.main_gui.header, len(events), self.event_type['size_path'])
        self.event_widget.load_header()

class AppendEvent(QUndoCommand):
    """ Append a new event. """

    def __init__(self, event_widget, event_type):
        super().__init__()
        self.event_widget = event_widget
        self.event_type = event_type

    def redo(self):
        """ Appends a new event to the end of the list. """
        project = self.event_widget.main_gui.project
        datatype = self.event_type['datatype']
        events = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])
        context = self.event_type['events_path'] + [len(events)]
        parents = properties.get_parents_by_path(self.event_widget.main_gui.header, context)
        events.append(project.model[datatype](project, context, parents))
        properties.set_member_by_path(self.event_widget.main_gui.header, len(events), self.event_type['size_path'])
        self.event_widget.load_header()

    def undo(self):
        """ Removes the last event. """
        events = properties.get_member_by_path(self.event_widget.main_gui.header, self.event_type['events_path'])
        events.pop()
        properties.set_member_by_path(self.event_widget.main_gui.header, len(events), self.event_type['size_path'])
        self.event_widget.load_header()

        