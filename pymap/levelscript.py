import agb.types
import backend

class LevelscriptTypeVar(agb.types.Structure):
    """ Class to encapsulate a levelscript that
    triggers only with a variable set equal to
    a value.
    This basically wraps arround a structure as
    the only real member of this type is the structure
    pointer itself.
    """

    def __init__(self):
        """ Initializes the levelscript var type. """
        super().__init__([
            ('var', agb.types.ScalarType('u16', 'vars')),
            ('value', agb.types.u16),
            ('script', backend.OwScriptPointerType()),
            ('field_8', agb.types.u16),
            ('field_A', agb.types.u16)
        ])

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves an levelscript based on var from rom.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
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
        levelscript : dict
            The levelscript with variables.
        offset : int
            The offeset of the next bytes after the levelscript structure was processed
        """
        # Retrieve the pointer to the struct
        struct_offset, offset = agb.types.pointer.from_data(rom, offset, project, context, parents)
        levelscript, _ = super().from_data(struct_offset, offset, project, context, parents)
        return levelscript, offset
    
    def to_assembly(self, event_header, parents, label=None, alignment=2, global_label=False):
        """ Creates an assembly representation of the levelscript with var type.
        
        Parameters:
        -----------
        levelscript : dict
            The levelscript.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        label : string or None
            The label to export (only if not None).
        alignment : int or None
            The alignment of the structure if required
        global_label : bool
            If the label generated will be exported globally.
            Only relevant if label is not None.

        Returns:
        --------
        assembly : str
            The assembly representation of the levelscript.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        