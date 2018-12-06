from agb.model.type import Type, associate_with_constant, label_and_align

class UnboundedArrayType(Type):
    """ Class for arrays with a variable length. It is terminated by some particular tail instance."""

    def __init__(self, datatype, tail):
        """ Initializes the array type of variable length.
        
        Parameters:
        -----------
        datatype : str
            The datatype the pointer is pointing to.
        tail : object
            An instanciation of the datatype that serves as tail for the array.   
        """
        self.datatype = datatype
        self.tail = tail

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the array type from a rom and tries to associate the constant value.
        
        Parameters:
        -----------
        rom : bytearray
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
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
        values : list
            A list of values in the array.
        """
        values = []
        parents = parents + [values]
        datatype = project.model[self.datatype]
        while True:
            value = datatype.from_data(rom, offset, project, context + [i], parents)
            if value == tail:
                break
            values.append(value)
        return values
    
    
    def to_assembly(self, values, project, context, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the union type.
        
        Parameters:
        -----------
        values : list
            The values of the array
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
            The assembly representation of the array.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        blocks = []
        additional_blocks = []
        parents = parents + [values]
        datatype = project.model[self.datatype]
        for i, value in enumerate(values + [self.tail]):
            block, additional = datatype.to_assembly(value, project, context + [i], parents)
            blocks.append(block)
            additional_blocks += additional
        assembly = '\n'.join(blocks)
        return label_and_align(assembly, label, alignment, global_label), additional_blocks

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
        values : list
            List with default initialized elements.
        """
        return []

    def size(self, values, project, context, parents):
        """ Returns the size of a specific structure instanze in bytes.

        Parameters:
        -----------
        values : list
            Elements of the array type.
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
        parents = parents + [values]
        datatype = project.model[self.datatype]
        for i, value in enumerate(values + [self.tail]):
            # Sum each element individually (this is more clean...)
            size += datatype.size(value, project, context + [i], parents)
        return size

    def get_constants(self, values, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        values : list
            Elements of the array type.
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
        parents = parents + [values]
        datatype = project.model[self.datatype]
        # Only using the first element would be faster, but this approach
        # is more clean and versatile.
        for i, value in enumerate(values):
            constants.update(datatype.get_constants(value, project, context + [i], parents))
        return constants
