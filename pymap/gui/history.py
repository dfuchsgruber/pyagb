from abc import ABC
import properties
from warnings import warn
from deepdiff import DeepDiff
import numpy as np

DEFAULT_HISTORY_CAPACITY = np.inf

class History:
    """ Module to encapsulate the history mechanisms of the gui. """
    def __init__(self, main_gui, capacity=DEFAULT_HISTORY_CAPACITY):
        self.main_gui = main_gui
        self.capcity = capacity
        self.clear()

    def add_action(self, action, aggregate=True):
        """ Adds a new action and aggregates actions of same type. """
        self.action_sets = self.action_sets[ : self.current_idx] # Clear actions of the hold history time line
        if self.current_idx > 0 and aggregate:
            # Check if the actions can be added to the last action group
            closed, actions = self.action_sets[self.current_idx - 1]
            if not closed:
                return actions.append(action)
        # Create a new action group
        self.action_sets.append([False, [action]])
        self.current_idx += 1
        # Truncate the history if necessary
        if len(self.action_sets) > self.capcity:
            self.action_sets = self.action_sets[1:]
            self.current_idx -= 1
            
    def do(self, action, aggregate=True):
        """ Performs a new action and adds it to the history. """
        self.add_action(action, aggregate=aggregate)
        action.do(self.main_gui)
    
    def undo(self):
        """ Undoes the last action group. """
        if self.current_idx > 0:
            for action in reversed(self.action_sets[self.current_idx - 1][1]):
                action.undo(self.main_gui)
            self.current_idx -= 1
    
    def redo(self):
        """ Redoes the last action group. """
        if self.current_idx < len(self.action_sets):
            for action in self.action_sets[self.current_idx][1]:
                action.do(self.main_gui)
            self.current_idx += 1
    
    def close(self):
        """ Closes the last performed action. """
        if self.current_idx > 0:
            self.action_sets[self.current_idx - 1][0] = True

    def clear(self):
        """ Clears the entire history. """
        self.action_sets = []
        self.current_idx = 0 # Points to the next redoable action
        self.save()

    def save(self):
        """ Changes the last saved state to the current state. """
        self.saved_idx = self.current_idx

    def is_unsaved(self):
        """ Checks if the history's current state is unsaved. """
        return self.saved_idx != self.current_idx

class ActionResize:
    """ Resizes a set of blocks. """

    def __init__(self, height_new, width_new, values_old):
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

class ActionResizeMap(ActionResize):
    """ Action for resizing the map blocks. """

    def _change_size(self, main_gui, blocks):
        properties.set_member_by_path(main_gui.footer, blocks.shape[0], main_gui.project.config['pymap']['footer']['map_height_path'])
        properties.set_member_by_path(main_gui.footer, blocks.shape[1], main_gui.project.config['pymap']['footer']['map_width_path'])
        properties.set_member_by_path(main_gui.footer, blocks, main_gui.project.config['pymap']['footer']['map_blocks_path'])
        main_gui.map_widget.load_header()

    def do(self, main_gui):
        """ Resizes the map blocks. """
        self._change_size(main_gui, self._new_blocks())

    def undo(self, main_gui):
        """ Undoes resizing of the map blocks. """
        self._change_size(main_gui, self._old_blocks())

class ActionResizeBorder(ActionResize):
    """ Action for resizing the map border. """

    def _change_size(self, main_gui, blocks):
        properties.set_member_by_path(main_gui.footer, blocks.shape[0], main_gui.project.config['pymap']['footer']['border_height_path'])
        properties.set_member_by_path(main_gui.footer, blocks.shape[1], main_gui.project.config['pymap']['footer']['border_width_path'])
        properties.set_member_by_path(main_gui.footer, blocks, main_gui.project.config['pymap']['footer']['border_path'])
        main_gui.map_widget.load_header()

    def do(self, main_gui):
        """ Resizes the map blocks. """
        self._change_size(main_gui, self._new_blocks())

    def undo(self, main_gui):
        """ Undoes resizing of the map blocks. """
        self._change_size(main_gui, self._old_blocks())

class ActionSetBorder:
    """ Action for setting border blocks. """

    def __init__(self, x, y, blocks_new, blocks_old):
        self.x = x
        self.y = y
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks, main_gui):
        # Helper method for setting a set of blocks
        properties.get_member_by_path(
            main_gui.footer, 
            main_gui.project.config['pymap']['footer']['border_path'])[
                self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], 0] = blocks[:, :, 0]
        main_gui.map_widget.load_map()
        main_gui.map_widget.load_border()

    def do(self, main_gui):
        """ Performs the setting of border blocks. """
        self._set_blocks(self.blocks_new, main_gui)
    
    def undo(self, main_gui):
        """ Undos setting of border blocks. """
        self._set_blocks(self.blocks_old, main_gui)

class ActionSetBlocks:
    """ Action for setting blocks. """

    def __init__(self, x, y, layers, blocks_new, blocks_old):
        self.x = x
        self.y = y
        self.layers = layers
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks, main_gui):
        # Helper method for setting a set of blocks
        properties.get_member_by_path(
            main_gui.footer, 
            main_gui.project.config['pymap']['footer']['map_blocks_path'])[
                self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], self.layers] = blocks[:, :, self.layers]
        main_gui.map_widget.update_map(self.x, self.y, self.layers, blocks)

    def do(self, main_gui):
        """ Performs the setting of blocks. """
        self._set_blocks(self.blocks_new, main_gui)
    
    def undo(self, main_gui):
        """ Undos setting of blocks. """
        self._set_blocks(self.blocks_old, main_gui)

class ActionReplaceBlocks:
    """ Action for replacing a set of blocks with another. """

    def __init__(self, idx, layer, value_new, value_old):
        self.idx = idx
        self.layer = layer
        self.value_new = value_new
        self.value_old = value_old

    def do(self, main_gui):
        """ Performs the flood fill. """
        map_blocks = properties.get_member_by_path(
            main_gui.footer, 
            main_gui.project.config['pymap']['footer']['map_blocks_path'])
        map_blocks[:, :, self.layer][self.idx] = self.value_new
        main_gui.map_widget.load_map()

    def undo(self, main_gui):
        """ Performs the flood fill. """
        map_blocks = properties.get_member_by_path(
            main_gui.footer, 
            main_gui.project.config['pymap']['footer']['map_blocks_path'])
        map_blocks[:, :, self.layer][self.idx] = self.value_old
        main_gui.map_widget.load_map()

class StateHistory:
    """ Pseudo-History class that does not acutally contain a history but encapsulates functionality of
    triggering save prompts based on a state. """

    def __init__(self, get_state):
        self.get_state = get_state # Function to get the current state
        self.reset()

    def do(self): pass

    def undo(self): pass
    
    def redo(self): pass

    def reset(self):
        self.state = self.get_state()

    def is_unsaved(self):
        diff = DeepDiff(self.get_state(), self.state)
        return 'value_changes' in diff or 'type_changes' in diff
    
