from . import properties, render
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

class AssignFooter(QUndoCommand):
    """ Class for assigning a footer. """
    
    def __init__(self, main_gui, label_new, label_old):
        super().__init__()
        self.main_gui = main_gui
        self.label_new = label_new
        self.label_old = label_old

    def _assign(self, label):
        """ Helper for assigning a label. """
        if self.main_gui.project is None or self.main_gui.header is None: return
        self.main_gui.open_footer(label)
    
    def redo(self):
        """ Performs the tileset assignment. """
        self._assign(self.label_new)

    def undo(self):
        """ Undoes the tileset assignment. """
        self._assign(self.label_old)

class AssignGfx(QUndoCommand):
    """ Class for assigning a gfx to a tileset. """
    
    def __init__(self, main_gui, primary, label_new, label_old):
        super().__init__()
        self.main_gui = main_gui
        self.primary = primary
        self.label_new = label_new
        self.label_old = label_old

    def _assign(self, label):
        """ Helper for assigning a label. """
        if self.main_gui.project is None or self.main_gui.header is None or self.main_gui.footer is None or self.main_gui.tileset_primary is None or self.main_gui.tileset_secondary is None: return
        if self.primary:
            self.main_gui.open_gfxs(label_primary = label)
        else:
            self.main_gui.open_gfxs(label_secondary = label)
    
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

class ChangeConnectionProperty(QUndoCommand):
    """ Change a property of any vent. """

    def __init__(self, connection_widget, connection_idx, mirror_offset, statements_redo, statements_undo):
        super().__init__()
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        self.mirror_offset = mirror_offset
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """ Executes the redo statements. """
        root = properties.get_member_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])[self.connection_idx]
        for statement in self.statements_redo:
            exec(statement)
        self.connection_widget.update_connection(self.connection_idx, self.mirror_offset)

    def undo(self):
        """ Executes the redo statements. """
        root = properties.get_member_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])[self.connection_idx]
        for statement in self.statements_undo:
            exec(statement)
        self.connection_widget.update_connection(self.connection_idx, self.mirror_offset)

class AppendConnection(QUndoCommand):
    """ Append a new connection. """

    def __init__(self, connection_widget):
        super().__init__()
        self.connection_widget = connection_widget

    def redo(self):
        """ Appends a new event to the end of the list. """
        project = self.connection_widget.main_gui.project
        datatype = self.connection_widget.main_gui.project.config['pymap']['header']['connections']['datatype']
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        context = self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'] + [len(connections)]
        parents = properties.get_parents_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        connections.append(project.model[datatype](project, context, parents))
        properties.set_member_by_path(
            self.connection_widget.main_gui.header, len(connections), self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_size_path'])
        self.connection_widget.load_header()

    def undo(self):
        """ Removes the last event. """
        connections = properties.get_member_by_path(
            self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        connections.pop()
        properties.set_member_by_path(
            self.connection_widget.main_gui.header, len(connections), self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_size_path'])
        self.connection_widget.load_header()

class RemoveConnection(QUndoCommand):
    """ Remove a connection. """

    def __init__(self, connection_widget, connection_idx):
        super().__init__()
        self.connection_widget = connection_widget
        self.connection_idx = connection_idx
        project = self.connection_widget.main_gui.project
        self.connection = properties.get_member_by_path(self.connection_widget.main_gui.header, project.config['pymap']['header']['connections']['connections_path'])[self.connection_idx]
    
    def redo(self):
        """ Removes the connection from the connections. """
        connections = properties.get_member_by_path(self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        connections.pop(self.connection_idx)
        properties.set_member_by_path(
            self.connection_widget.main_gui.header, len(connections), self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_size_path'])
        self.connection_widget.load_header()

    def undo(self):
        """ Reinserts the connection. """
        connections = properties.get_member_by_path(self.connection_widget.main_gui.header, self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_path'])
        connections.insert(self.connection_idx, self.connection)
        properties.set_member_by_path(
            self.connection_widget.main_gui.header, len(connections), self.connection_widget.main_gui.project.config['pymap']['header']['connections']['connections_size_path'])
        self.connection_widget.load_header()

class ChangeBlockProperty(QUndoCommand):
    """ Change a property of any block. """

    def __init__(self, tileset_widget, block_idx, statements_redo, statements_undo):
        super().__init__()
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """ Executes the redo statements. """
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary']
        tileset = self.tileset_widget.main_gui.tileset_primary if self.block_idx < 0x280 else self.tileset_widget.main_gui.tileset_secondary
        root = properties.get_member_by_path(tileset, config['behaviours_path'])[self.block_idx % 0x280]
        for statement in self.statements_redo:
            exec(statement)
        self.tileset_widget.block_properties.update()

    def undo(self):
        """ Executes the redo statements. """
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary']
        tileset = self.tileset_widget.main_gui.tileset_primary if self.block_idx < 0x280 else self.tileset_widget.main_gui.tileset_secondary
        root = properties.get_member_by_path(tileset, config['behaviours_path'])[self.block_idx % 0x280]
        for statement in self.statements_undo:
            exec(statement)
        self.tileset_widget.block_properties.update()

class SetTiles(QUndoCommand):
    """ Changes the tiles of any block. """

    def __init__(self, tileset_widget, block_idx, layer, x, y, tiles_new, tiles_old):
        super().__init__()
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx
        self.layer = layer
        self.x = x
        self.y = y
        self.tiles_new = tiles_new
        self.tiles_old = tiles_old

    def _set_tiles(self, tiles):
        """ Helper method to set tiles. """
        tileset = self.tileset_widget.main_gui.tileset_primary if self.block_idx < 0x280 else self.tileset_widget.main_gui.tileset_secondary
        path = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary']['blocks_path']
        blocks = properties.get_member_by_path(tileset, path)
        block = np.array(blocks[self.tileset_widget.selected_block % 0x280]).reshape(3, 2, 2)
        block[self.layer, self.y : self.y + tiles.shape[0], self.x : self.x + tiles.shape[1]] = tiles
        blocks[self.block_idx % 0x280] = block.flatten().tolist()
        # Update the block
        self.tileset_widget.main_gui.blocks[self.block_idx] = render.get_block(blocks[self.block_idx % 0x280], self.tileset_widget.main_gui.tiles, self.tileset_widget.main_gui.project)
        self.tileset_widget.load_blocks()
        if self.layer == 0: self.tileset_widget.block_lower_scene.update_block()
        elif self.layer == 1: self.tileset_widget.block_mid_scene.update_block()
        elif self.layer == 2: self.tileset_widget.block_upper_scene.update_block()

    def redo(self):
        """ Performs the change on the block. """
        self._set_tiles(self.tiles_new)

    def undo(self):
        """ Undoes the change on the block. """
        self._set_tiles(self.tiles_old)

class SetPalette(QUndoCommand):
    """ Changes a palette of a tileset. """

    def __init__(self, tileset_widget, pal_idx, palette_new, palette_old):
        super().__init__()
        self.tileset_widget = tileset_widget
        self.pal_idx = pal_idx
        self.palette_new = palette_new
        self.palette_old = palette_old

    def _set_palette(self, palette):
        """ Helper method to set a palette. """
        if self.pal_idx < 7:
            palettes = properties.get_member_by_path(self.tileset_widget.main_gui.tileset_primary, self.tileset_widget.main_gui.project.config['pymap']['tileset_primary']['palettes_path'])
        else:
            palettes = properties.get_member_by_path(self.tileset_widget.main_gui.tileset_secondary, self.tileset_widget.main_gui.project.config['pymap']['tileset_secondary']['palettes_path'])
        palettes[self.pal_idx % 7] = palette
        # Update tiles and blocks
        self.tileset_widget.main_gui.load_blocks()
        self.tileset_widget.reload()

    def redo(self):
        """ Sets the new palette. """
        self._set_palette(self.palette_new)

    def undo(self):
        """ Undoes the setting of the new palette. """
        self._set_palette(self.palette_old)



def path_to_statement(path, old_value, new_value):
    """ Transforms a path to a property into a redoable statement relative to a 'root' instance. """
    path = ''.join(map(lambda member: f'[{repr(member)}]', path))
    return f'root{path} = {repr(str(new_value))}', f'root{path} = {repr(str(old_value))}'

class SetTilesetAnimation(QUndoCommand):
    """ Changes the animation of a tileset. """
    
    def __init__(self, tileset_widget, primary, value_new):
        super().__init__()
        self.tileset_widget = tileset_widget
        self.primary = primary
        self.value_new = value_new
        widget = self.tileset_widget.animation_primary_line_edit if self.primary else self.tileset_widget.animation_secondary_line_edit
        tileset = self.tileset_widget.main_gui.tileset_primary if self.primary else self.tileset_widget.main_gui.tileset_secondary
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.primary else 'tileset_secondary']
        self.value_old = properties.get_member_by_path(tileset, config['animation_path'])

    def _set_value(self, value):
        """ Helper method to set a value to the custom LineEdit. """
        widget = self.tileset_widget.animation_primary_line_edit if self.primary else self.tileset_widget.animation_secondary_line_edit
        tileset = self.tileset_widget.main_gui.tileset_primary if self.primary else self.tileset_widget.main_gui.tileset_secondary
        config = self.tileset_widget.main_gui.project.config['pymap']['tileset_primary' if self.primary else 'tileset_secondary']
        widget.blockSignals(True)
        widget.setText(str(value))
        widget.blockSignals(False)
        properties.set_member_by_path(tileset, str(value), config['animation_path'])

    def redo(self):
        """ Sets the new value. """
        self._set_value(self.value_new)

    def undo(self):
        """ Undoes the setting of the new value. """
        self._set_value(self.value_old)

class AppendAutoShape(QUndoCommand):
    """ Adds an automatic shape. """

    def __init__(self, main_gui):
        super().__init__()
        self.main_gui = main_gui

    def redo(self):
        """ Resizes the map blocks. """
        self.main_gui.project.automatic_shapes.append([0 for _ in range(15)])
        self.main_gui.map_widget.load_auto()

    def undo(self):
        """ Undoes resizing of the map blocks. """
        self.main_gui.project.automatic_shapes.pop(0)
        self.main_gui.map_widget.load_auto()

class RemoveAutoShape(QUndoCommand):
    """ Removes an automatic shape. """

    def __init__(self, main_gui, idx=-1):
        super().__init__()
        self.main_gui = main_gui
        self.idx = idx
        self.shape = self.main_gui.project.automatic_shapes[self.idx]
    
    def redo(self):
        """ Removes the event from the events. """
        self.main_gui.project.automatic_shapes.pop(self.idx)
        self.main_gui.map_widget.load_auto()

    def undo(self):
        """ Reinserts the event. """
        self.main_gui.project.automatic_shapes.insert(self.idx, self.shape)
        self.main_gui.map_widget.load_auto()