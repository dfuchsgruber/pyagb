from abc import ABC
import properties

DEFAULT_HISTORY_CAPACITY = 0x10000
ACTION_SET_BLOCKS = 0

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


class ActionSetBlocks:
    
    action_type = ACTION_SET_BLOCKS

    """ Action for setting blocks. """
    def __init__(self, x, y, blocks_new, blocks_old):
        self.x = x
        self.y = y
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks, main_gui):
        # Helper method for setting a set of blocks
        properties.get_member_by_path(
            main_gui.footer, 
            main_gui.project.config['pymap']['footer']['map_blocks_path'])[
                self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1]] = blocks
        main_gui.map_widget.update_map(self.x, self.y, blocks)

    def do(self, main_gui):
        """ Performs the setting of blocks. """
        self._set_blocks(self.blocks_new, main_gui)
    
    def undo(self, main_gui):
        """ Undos setting of blocks. """
        self._set_blocks(self.blocks_old, main_gui)
