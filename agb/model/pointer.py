from agb.model.scalar_type import ScalarType
from agb.model.type import Type, associate_with_constant, label_and_align

class PointerType(ScalarType):
    """ Class to models pointers. """

    def __init__(self, datatype, label_get):
        """ Initializes the pointer to another datatype.
        
        Parameters:
        -----------
        datatype : str
            The datatype the pointer is pointing to.
        label_get : function
            Function that creates the label for the structure.
            
            Parameters:
            -----------
            project : pymap.project.Project
                The map project
            context : list
                Context from parent elements.
            parents : list
                The parent values of this value. The last
                element is the direct parent. The parents are
                possibly not fully initialized as the values
                are explored depth-first.

            Returns:
            --------
            label : str or None
                The label of the data the pointer points to.
            alignment : int or None
                The alignment of the data that the pointer points to.
            global_label : bool
                If the data's label will be exported globally.
            """
        super().__init__('pointer', constant=None)
        self.datatype = datatype
        self.label_get = label_get

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the pointer type from a rom.
        
        Parameters:
        -----------
        rom : bytearray
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the pointer from.
        proj : pymap.project.Project
            The pymap project to access e.g. constants.
        context : list of str
            The context in which the data got initialized.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        
        Returns:
        --------
        data : object
            The data the pointer is pointing to.
        """
        data_offset =  super().from_data(rom, offset, project, context, parents)
        if data_offset is None:
            # Nullpointer
            return None
        # Retrieve the data
        datatype = project.model[self.datatype]
        return datatype.from_data(rom, data_offset, project, context, parents)
    
    def to_assembly(self, data, project, context, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the pointer type.
        
        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
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
            The assembly representation of the pointer.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        if data is None:
            # Nullpointer
            return super().to_assembly(data, project, context, parents, label=label, alignment=alignment, global_label=global_label)
        
        data_label, data_alignment, data_global_label = self.label_get(project, context, parents)
        assembly, additional_blocks = super().to_assembly(data_label, project, context, parents, label=label, alignment=alignment, global_label=global_label)
        # Create assembly for the datatype that the pointer refers to
        datatype = project.model[self.datatype]
        data_assembly, data_additional_blocks = datatype.to_assembly(data, project, context, parents, label=data_label, alignment=data_alignment, global_label=data_global_label)
        # The data assembly is an additional block as well
        additional_blocks.append(data_assembly)
        additional_blocks += data_additional_blocks
        return assembly, additional_blocks

    def __call__(self, project, context, parents):
        """ Initializes a new array. 
        
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
        data : object
            Default intialized object.
        """
        datatype = project.model[self.datatype]
        return datatype(project, context, parents)


    def get_constants(self, data, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
        datatype = project.model[self.datatype]
        if data is None:
            return set()
        return datatype.get_constants(data, project, context, parents)
