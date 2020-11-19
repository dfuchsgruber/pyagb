from agb.model.type import Type, associate_with_constant, label_and_align

class UnionType(Type):
    """ Class for union type. """

    def __init__(self, subtypes, name_get):
        """ Initializes the union type.
        
        Parameters:
        -----------
        subtypes : dict
            Maps str -> str, i.e. the name of a subtype to the
            respective subtype identifier.
        name_get : function
            Function that returns the name of the subtype that is currently
            used for the union. The union type will be exported to assembly
            for this type and only values for this subtype will be retrieved
            considering the from_data function. Therefore values of other
            subtypes remain default initialized.
            
            Parameters:
            -----------
            project : pymap.project.Project
                The pymap project.
            context : list
                Context from parent elements.
            parents : list
                The parent values of this value. The last
                element is the direct parent. The parents are
                possibly not fully initialized as the values
                are explored depth-first.

            Returns:
            --------
            active_subtype : str
                The name of the subtype that is active in the union.
        """
        self.subtypes = subtypes
        self.name_get = name_get
    
    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the constant type from a rom and tries to associate the constant value.
        
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
        values : dict
            Dict that maps from the subtype names to their value instanciation.
        """
        values = {}
        if self.name_get(project, context, parents) not in self.subtypes:
            raise RuntimeError(f'Active subtype of union {self.name_get(project, context, parents)} not part of union type.')

        for name in self.subtypes:
            subtype = project.model[self.subtypes[name]]
            if name == self.name_get(project, context, parents):
                value = subtype.from_data(rom, offset, project, context + [name], parents + [values])
                values[name] = value
            else:
                values[name] = subtype(project, context, parents)

        return values

    def to_assembly(self, values, project, context, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the union type.
        
        Parameters:
        -----------
        values : dict
            Dict that maps from the subtype names to their value instanciaton.
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
            The assembly representation of the specific subtype.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        active_subtype_name = self.name_get(project, context, parents)
        active_subtype = project.model[self.subtypes[active_subtype_name]]
        return active_subtype.to_assembly(values[active_subtype_name], project, context + [active_subtype_name], parents + [values], label=label, alignment=alignment, global_label=global_label)

    def __call__(self, project, context, parents):
        """ Initializes a new empty union type. 
        
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
        value : dict
            Dict that maps from each subtype to the default initializaiton.
        """
        value = {}
        for name in self.subtypes:
            value[name] = project.model[self.subtypes[name]](project, context + [name], parents + [value])
        return value

    def size(self, values, project, context, parents):
        """ Returns the size of a specific structure instanze in bytes.

        Parameters:
        -----------
        values : dict
            Dict that maps from the subtype names to their value instanciaton.
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
        subtype_name = self.name_get(project, context, parents)
        subtype = project.model[self.subtypes[subtype_name]]
        return subtype.size(values[subtype_name], project, context + [subtype_name], parents + [values])

    def get_constants(self, values, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        values : dict
            Dict that maps from the subtype names to their value instanciaton.
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
        subtype_name = self.name_get(project, context, parents)
        subtype = project.model[self.subtypes[subtype_name]]
        return subtype.get_constants(values[subtype_name], project, context + [subtype_name], parents + [values])
     