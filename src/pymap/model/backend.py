"""Model for types that can export additional assets when generated from data.

Each of these types triggers a callback when exported, i.e. when `from_data`
is called. These should be extended in your project's config file to trigger
the desired behaviour. E.g., one can trigger the export of png files when
gfx pointers are exported.
"""
from __future__ import annotations
from typing import Callable

from agb.model.type import Model, ModelContext, ModelParents, ModelValue
from agb.model.scalar_type import ScalarType
from pymap.project import Project


def ow_script(rom: bytearray, offset: int, project: Project, context: ModelContext,
              parents: ModelParents) -> str:
    """Callback for exporting overworld scripts.

    Backend for exporting overworld scripts.
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


def gfx(rom: bytearray, offset: int, project: Project, context: ModelContext,
        parents: ModelParents, lz77_compressed: bool) -> str:
    """Callback that is called when a gfx is exported.

    Backend for exporting gfxs.
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

def tileset(rom: bytearray, offset: int, project: Project, context: ModelContext,
            parents: ModelParents) -> str:
    """Callback that is called when a tileset is exported.

    Backend for exporting tilesets.
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

def levelscript_header(rom: bytearray, offset: int, project: Project,
                       context: ModelContext, parents: ModelParents) -> str:
    """Callback that is called when a levelscript header is exported.

    Backend for exporting levelscript header.
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

def footer(rom: bytearray, offset: int, project: Project, context: ModelContext,
           parents: ModelParents) -> str:
    """Callback that is called when a map footer is exported.

    Backend for exporting map footer.
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


class BackendPointerType(ScalarType):
    """Class for pointers that invoke a callback when exported.

    This class is a wrapper around the pointer type that invokes
    a callback when the pointer is exported from data using the
    `from_data` method.
    """

    def __init__(self,
                 export: Callable[[bytearray, int, Project,
                                   ModelContext, ModelParents], str]):
        """Initializes the abstract pointer type.

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

    def from_data(self, rom: bytearray, offset: int, project: Project,
                  context: ModelContext, parents: ModelParents) -> ModelValue:
        """Retrieves the overworld script pointer type from a rom.

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
        if value is None:
            return '0'
        assert isinstance(value, int), f"Expected an int, got {value}"
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
default_model: Model = {
    'ow_script_pointer' : ow_script_pointer_type,
    'tileset_pointer' : tileset_pointer_type,
    'footer_pointer' : footer_pointer_type,
    'levelscript_header_pointer' : levelscript_header_type
}
