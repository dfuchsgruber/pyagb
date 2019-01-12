from agb.model.type import Type, associate_with_constant, label_and_align

class Structure(Type):
    """ Superclass to model any kind of structure """

    def __init__(self, structure, hidden_members=set()):
        """ Initialize the members with priorized members. 
        
        Parameters:
        -----------
        structure : list of tuples (member, type)
            member : str
                The name of the member
            type : str
                The name of the type
            priority : int
                Members with lower priority will get processed first
        
        hidden_members : set
            The members to hide when displaying the structure in a parameter tree.
        """
        self.structure = structure
        self.hidden_members = hidden_members

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the structure from a rom.

        Parameters:
        -----------
        rom : bytearray
            The rom to initialize the structure from
        offset : int
            The offset to initialize the structure from
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list of str
            The context in which the data got initialized
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.

        Returns:
        --------
        structure : dict
            The initialized structure
        """
        structure = {}
        parents = parents + [structure]

        export_closures = {} # Store functions that will export the members here
        
        # Create callbacks in the order of members
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            def export_member():
                # Helper function that will export the member
                value = datatype.from_data(rom, offset, project, context + [attribute], parents)
                structure[attribute] = value
            export_closures[attribute] = export_member
            offset += datatype.size(structure[attribute], project, parents)
        
        # Export members w.r.t. to their priority
        for attribute, datatype_name, _ in sorted(self.structure, key=lambda x: x[2]):
            export_closures[attribute]()

        return structure

    def __call__(self, project, context, parents):
        """ Initializes a new structure with default values. 
        
        Parameters:
        -----------
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.
        
        Returns:
        --------
        structure : dict
            New structure with default values.
        """
        structure = {}
        parents = parents + [structure]
        # Initialize priorized members first
        for attribute, datatype_name, _ in sorted(self.structure, key=lambda x: x[2]):
            structure[attribute] = datatype(project, context + [attribute], parents)
        return structure


    def to_assembly(self, structure, project, context, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly representation of a structure.

        structure : dict
            The structure to convert to an assembly string.
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent.
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
            The assembly representation.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        assemblies, additional_blocks = [], []
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            assembly_datatype, additional_blocks_datatype = datatype.to_assembly(structure[attribute], project, context + [attribute], parents + [structure])
            assemblies.append(assembly_datatype)
            additional_blocks += additional_blocks_datatype
        
        return label_and_align('\n'.join(assemblies), label, alignment, global_label), additional_blocks

    def size(self, structure, project, context, parents):
        """ Returns the size of a specific structure instanze in bytes.

        Parameters:
        -----------
        structure : dict
            The structure of which the size is desired.
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.
        
        Returns:
        --------
        length : int
            The size of this type in bytes.
        """
        size = 0
        parents = parents + [structure]
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            size += datatype.size(structure[attribute], project, context + [attribute], parents)
        return size
    
    def get_constants(self, structure, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        structure : dict
            The structure of which the size is desired.
        project : pymap.project.Project
            The project to e.g. fetch constants from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are generated depth-first.
        
        Returns:
        --------
        constants : set of str
            A set of all required constants.
        """
        constants = set()
        parents = parents + [structure]
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            constants.update(datatype.get_constants(structure[attribute], project, context + [attribute], parents))
        return constants
    
