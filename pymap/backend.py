# Module that provides backend for various exporting and displaying stuff
# Variables are changed during initialization of the project

import agb.types

def ow_script(rom, offset, project, context):
    """ Backend for exporting overworld scripts.
    This function is called everytime a ow_script
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom instance to export scripts from
    offset : int
        The offset to export the script from
    project : pymap.project.Project
        The pymap project
    context : list of str
        The context in which the data got initialized

    Returns:
    --------
    label : str
        The label associated with the script offset
    """
    print(f'Encoutered ow_script @{hex(offset)} in context {context}')
    return hex(offset)


def gfx(rom, offset, project, context, compressed):
    """ Backend for exporting gfx's.
    This function is called everytime a ow_script
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom instance to export scripts from
    offset : int
        The offset to export the script from
    project : pymap.project.Project
        The pymap project
    context : list of str
        The context in which the data got initialized
    compressed : bool
        If the gfx is compressed.

    Returns:
    --------
    label : str
        The label associated with the gfx offset
    """
    print(f'Encoutered gfx @{hex(offset)} in context {context}')
    return hex(offset)

class OwScriptPointerType(agb.types.ScalarType):
    """ Type class for overworld script pointers """

    def __init__(self):
        super().__init__('pointer')

    def from_data(self, rom, offset, project, context):
        """ Retrieves the overworld script pointer type from a rom.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to retrieve the data from.
        offset : int
            The offset of the pointer to an overworld script.
        project : pymap.project.Project
            The pymap project to access e.g. constants.
        context : list of str
            The context in which the data got initialized
        
        Returns:
        --------
        label : str
            The label associated with the overworld script
        offset : int
            The offeset of the next bytes after the value was processed
        """
        value, offset = super().from_data(rom, offset, project, context)
        if value is None: return '0', offset
        label = ow_script(rom, value, project, context)
        return label, offset

class GfxPointerType(agb.types.ScalarType):
    """ Type class for overworld script pointers """

    def __init__(self, compressed):
        """ Initializes the pointer to gfx type. 
        
        Parameters:
        -----------
        compressed : bool
            If the gfx the pointer points to is lz77 compressed. """
        super().__init__('pointer')
        self.compressed = compressed

    def from_data(self, rom, offset, project, context):
        """ Retrieves the overworld script pointer type from a rom.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to retrieve the data from.
        offset : int
            The offset of the pointer to an overworld script.
        project : pymap.project.Project
            The pymap project to access e.g. constants.
        context : list of str
            The context in which the data got initialized
        
        Returns:
        --------
        label : str
            The label associated with the overworld script
        offset : int
            The offeset of the next bytes after the value was processed
        """
        value, offset = super().from_data(rom, offset, project, context)
        if value is None: return '0', offset
        label = gfx(rom, value, project, self.compressed, context)
        return label, offset