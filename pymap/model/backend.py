# Module that provides backend for various exporting and displaying stuff
# Variables are changed during initialization of the project

import agb.types

def ow_script(rom, offset, project, context, parents):
    """ Backend for exporting overworld scripts.
    This function is called everytime a ow_script
    type is exported from data.

    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom to retrieve the data from.
    offset : int
        The pointer to the abstract type.
    project : pymap.project.Project
        The pymap project to access e.g. constants.
    context : list of str
        The context in which the data got initialized
    parents : list
        The parent values of this value. The last
        element is the direct parent. The parents are
        possibly not fully initialized as the values
        are explored depth-first.
    
    Returns:
    --------
    label : str
        The label assoicated with the script.
    """
    print(f'Encoutered ow_script @{hex(offset)} in context {context}')
    return hex(offset + 0x8000000)


def gfx(rom, offset, project, context, parents, lz77_compressed):
    """ Backend for exporting gfxs.
    This function is called everytime a gfx
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom to retrieve the data from.
    offset : int
        The pointer to the abstract type.
    project : pymap.project.Project
        The pymap project to access e.g. constants.
    context : list of str
        The context in which the data got initialized
    parents : list
        The parent values of this value. The last
        element is the direct parent. The parents are
        possibly not fully initialized as the values
        are explored depth-first.
    lz77_compressed : bool
        If the data is lz77 compressed.
    
    Returns:
    --------
    label : str
        The label assoicated with the gfx.
    """
    print(f'Encoutered gfx @{hex(offset)} in context {context}')
    return hex(offset + 0x8000000)

def tileset(rom, offset, project, context, parents):
    """ Backend for exporting tilesets.
    This function is called everytime a tileset
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom to retrieve the data from.
    offset : int
        The pointer to the abstract type.
    project : pymap.project.Project
        The pymap project to access e.g. constants.
    context : list of str
        The context in which the data got initialized
    parents : list
        The parent values of this value. The last
        element is the direct parent. The parents are
        possibly not fully initialized as the values
        are explored depth-first.
    lz77_compressed : bool
        If the data is lz77 compressed.
    
    Returns:
    --------
    label : str
        The label assoicated with the tileset.
    """
    print(f'Encoutered tileset @{hex(offset)} in context {context}')
    return hex(offset + 0x08000000)

def levelscript_header(rom, offset, project, context, parents):
    """ Backend for exporting levelscript header.
    This function is called everytime a levelscript header
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom to retrieve the data from.
    offset : int
        The pointer to the abstract type.
    project : pymap.project.Project
        The pymap project to access e.g. constants.
    context : list of str
        The context in which the data got initialized
    parents : list
        The parent values of this value. The last
        element is the direct parent. The parents are
        possibly not fully initialized as the values
        are explored depth-first.
    lz77_compressed : bool
        If the data is lz77 compressed.
    
    Returns:
    --------
    label : str
        The label assoicated with the levescript header.
    """
    print(f'Encoutered levelscript header @{hex(offset)} in context {context}')
    return hex(offset + 0x08000000)

def footer(rom, offset, project, context, parents):
    """ Backend for exporting map footer.
    This function is called everytime a map footer
    type is exported from data.
    
    Parameters:
    -----------
    rom : agb.agbrom.Agbrom
        The rom to retrieve the data from.
    offset : int
        The pointer to the abstract type.
    project : pymap.project.Project
        The pymap project to access e.g. constants.
    context : list of str
        The context in which the data got initialized
    parents : list
        The parent values of this value. The last
        element is the direct parent. The parents are
        possibly not fully initialized as the values
        are explored depth-first.
    lz77_compressed : bool
        If the data is lz77 compressed.
    
    Returns:
    --------
    label : str
        The label assoicated with the map footer.
    """
    print(f'Encoutered map footer @{hex(offset)} in context {context}')
    return hex(offset + 0x08000000)


class BackendPointerType(agb.types.ScalarType):
    """ Class for pointers that represent structures that
    may or may not be exported by the backend of pymap."""

    def __init__(self, export):
        """ Initializes the abstract pointer type.
        
        Parameters:
        -----------
        export : function
            Function that is executed when the type is
            exported.
            
            Parameters:
            -----------
            rom : agb.agbrom.Agbrom
                The rom to retrieve the data from.
            offset : int
                The pointer to the abstract type.
            project : pymap.project.Project
                The pymap project to access e.g. constants.
            context : list of str
                The context in which the data got initialized
            parents : list
                The parent values of this value. The last
                element is the direct parent. The parents are
                possibly not fully initialized as the values
                are explored depth-first.
            
            Returns:
            --------
            label : str
                The label assoicated with the type.
        """
        super().__init__('pointer')
        self.export = export

    def from_data(self, rom, offset, project, context, parents):
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
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        
        Returns:
        --------
        label : str
            The label associated with the overworld script
        """
        value = super().from_data(rom, offset, project, context, parents)
        if value is None: return '0'
        return self.export(rom, value, project, context, parents)

# Define a type for overworld scripts
ow_script_pointer_type = BackendPointerType(ow_script)

# Define a type for tilesets
tileset_pointer_type = BackendPointerType(tileset)

# Define a type for map footers
footer_pointer_type = BackendPointerType(footer)

# Define a type for levelscript headers
levelscript_header_type = BackendPointerType(levelscript_header)

# These models will be exported
default_model = {
    'ow_script_pointer' : ow_script_pointer_type,
    'tileset_pointer' : tileset_pointer_type,
    'footer_pointer' : footer_pointer_type,
    'levelscript_header_pointer' : levelscript_header_type
}