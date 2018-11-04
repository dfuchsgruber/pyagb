# Import and export functions for scalar types and non-scalar types

from warnings import warn
import struct
import agb.string.agbstring
import agb.string.compile

class Structure:
    """ Superclass to model any kind of structure """

    def __init__(self, structure, priorized_members=[]):
        """ Initialize the members with priorized members. 
        
        Parameters:
        -----------
        structure : list of tuples (member, type)
            member : str
                The name of the member
            type : str
                The name of the type
        
        priorized_members : list of str
            The members to export first (the first index will be exported first,
            the second afterwards and so on...)
        """
        self.structure = structure
        self.priorized_members = priorized_members

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
        # Export priorized members first
        for priorized_attribute in self.priorized_members:
            offset_priorized = offset
            for attribute, datatype_name in self.structure:
                datatype = project.model[datatype_name]
                if attribute == priorized_attribute:
                    # Export the member beforehand
                    value = datatype.from_data(rom, offset_priorized, project, context + [attribute], parents)
                    structure[attribute] = value
                else:
                    # Use an empty initiaization as stub
                    value = None
                offset_priorized += datatype.size(value, project, parents)
        
        # Export the other attributes with a fully initialized structure
        for attribute, datatype_name in self.structure:
            datatype = project.model[datatype_name]
            if attribute not in self.priorized_members:
                # Export other attribute
                value = datatype.from_data(rom, offset, project, context + [attribute], parents)
                structure[attribute] = value
            offset += datatype.size(structure[attribute], project, parents)
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
        for priorized_attribute in self.priorized_members:
            for attribute, datatype_name in self.structure:
                datatype = project.model[datatype_name]
                if attribute == priorized_attribute:
                    structure[attribute] = datatype(project, context + [attribute], parents)

        # Initialize other members
        for attribute, datatype_name in self.structure:
            if attribute not in self.priorized_members:
                datatype = project.model[datatype_name]
                structure[attribute] = datatype(project, context + [attribute], parents + [structure])
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
        for attribute, datatype_name in self.structure:
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
        for attribute, datatype_name in self.structure:
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
        for attribute, datatype_name in self.structure:
            datatype = project.model[datatype_name]
            constants.update(datatype.get_constants(structure[attribute], project, context + [attribute], parents))
        return constants
    



class ScalarType:
    """ Class to model scalar types. """

    def __init__(self, fmt, constant=None):
        """ Initailizes the scalar type.
        
        Parameters:
        -----------
        fmt : str
            The format string for the scalar type. Either (u|s)(8|16|32) or 'pointer'.
        constant : str or None
            The constant table associated with the scalar type.
        """
        self.fmt = fmt
        self.constant = constant

    def from_data(self, rom, offset, proj, context, parents):
        """ Retrieves the scalar type from a rom.
        
        Parameters:
        -----------
        rom : bytearray
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
        proj : pymap.project.Project
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
        value : int
            The value at the given offset in rom.
        """
        value = scalar_from_data[self.fmt](rom, offset, proj)

        # Try to associate the value with a constant
        value = associate_with_constant(value, proj, self.constant)
        return value

    def to_assembly(self, value, project, context, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        value : string or int
            The value to export
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
            The assembly representation.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        return label_and_align(scalar_to_assembly[self.fmt](value), label, alignment, global_label), []

    def __call__(self, project, context, parents):
        """ Returns a new empty value (0). 
        
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
        value : int
            Zero value (0)."""
        return 0

    def size(self, value, project, context, parents):
        """ Returns the size of a specific structure instanze in bytes.

        Parameters:
        -----------
        value : int or str
            The value of the scalar type
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
        return scalar_to_length[self.fmt]

    def get_constants(self, value, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        value : int or str
            The value of the scalar type
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
        if self.constant:
            return set([self.constant])
        else:
            return set()


class BitfieldType(ScalarType):
    """ Class for bitfield types. """
    
    def __init__(self, fmt, structure):
        """ Initializes the bitfield type. 
        
        Parameters:
        -----------
        fmt : str
            A string {s/u}{bitlength} that encodes the scalar type that underlies the bitfield.
            Example: 'u8', 's32', ...
        structure : list of triplets
            Define the structure of the bitfield. Each element consists of:
            member : str
                The name of the member.
            constant : str or None
                The type of the field
            size : int
                The number of bits this member spans
            Note that the size of the entire bitfield is infered from the structure
            attribute and padded to fit 8, 16 or 32 bit.
        """
        super().__init__(fmt, constant=None)
        self.structure = structure

    def from_data(self, rom, offset, proj, context, parents):
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
            The context in which the data got initialized
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.
        
        Returns:
        --------
        value : list of int or str
            The values at the given offset in rom associated with a constant string if possible.
        """
        bit_idx = 0
        scalar_value = scalar_from_data[self.fmt](rom, offset, proj)
        value = []
        for member, constant, size in self.structure:
            mask = (1 << size) - 1
            member_value = associate_with_constant((scalar_value >> bit_idx) & mask, proj, constant)
            value.append(member_value)
            bit_idx += size
        return value

    def to_assembly(self, values, project, context, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        values : list of string or int
            The values to export
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
            The assembly representation instruction line
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        if not len(values) == len(self.structure):
            raise RuntimeError(f'More values for bitfield provided {len(values)} than structure supports {len(self.structure)}')
        shifted = []
        bit_idx = 0
        for value, triplet in zip(values, self.structure):
            member, _, size = triplet # Unpack the member information
            mask = (1 << size) - 1
            shifted.append(f'(({value} & {mask}) << {bit_idx})')
            bit_idx += size
        assembly = scalar_to_assembly[self.fmt](' | '.join(shifted))
        return label_and_align(assembly, label, alignment, global_label), []

    def __call__(self, project, context, parents):
        """ Initializes a new empty bitfield.
        
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
        value : list of int or str
            The empty bitfield (zeros).
        """
        return [0] * len(self.structure)

    def get_constants(self, values, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        values : list of string or int
            The values to export
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
        return set([
            constant for member, constant, size in self.structure if constant is not None
        ])

class UnionType:
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
                values[name] = subtype(project, parents)

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
            value[name] = project.model[self.subtypes[names]](project, context + [name], parents + [value])
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
        
class ArrayType:
    """ Type for arrays. """

    def __init__(self, datatype, size_get):
        """ Initializes the array type.
        
        Parameters:
        -----------
        datatype : str
            The underlying datatype
        size_get : function
            Function that returns the number of elements in the array. This
            function can access all parents of the array type, but not
            all parent's elements will be initialized, as the exporting
            of types is depth-first.
            Parameters:
            -----------
            project : pymap.project
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
            size : int
                The number of elements in the array.
        """
        self.datatype = datatype
        self.size_get = size_get

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
        values : list
            A list of values in the array.
        """
        num_elements = self.size_get(project, context, parents)
        values = []
        datatype = project.model[self.datatype]
        for i in range(num_elements):
            value = datatype.from_data(rom, offset, project, context + [i], parents + [values])
            values.append(value)
            offset += datatype.size(value, project, context + [i], parents + [values])
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
        datatype = project.model[self.datatype]
        for i in range(self.size_get(project, context, parents)):
            block, additional = datatype.to_assembly(values[i], project, context + [i], parents + [values])
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
        values = []
        datatype = project.model[self.datatype]
        for i in range(self.size_get(project, context, parents)):
            values.append(datatype(project, context + [i], parents + [values]))
        return values

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
        num_elements = self.size_get(project, context, parents)
        size = 0
        parents = parents + [values]
        datatype = project.model[self.datatype]
        for i in range(num_elements):
            # Sum each element individually (this is more clean...)
            size += datatype.size(values[i], project, context + [i], parents)
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
        num_elements = self.size_get(project, context, parents)
        constants = set()
        parents = parents + [values]
        datatype = project.model[self.datatype]
        # Only using the first element would be faster, but this approach
        # is more clean and versatile.
        for i in range(num_elements):
            constants.update(datatype.get_constants(values[i], project, context + [i], parents))
        return constants

class VariableLengthArrayType:
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
        for i, value in enumerate(values + [self.tail]):
            constants.update(datatype.get_constants(value, project, context + [i], parents))
        return constants

    

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


class StringType:
    """ Type class for strings. """

    def __init__(self, fixed_size=None, box_size=None):
        """ Initializes a string type.
        
        Parameters:
        -----------
        charmap : str
            Path to the character map.
        fixed_size : int or None
            The fixed size of the string in bytes. Zeros
            will be padded when compiling. If None, the
            string has variable size.
            Note that this argument is incompatible with the box_size
            argument.
        box_size : tuple or None
            A tuple width, height indicating the width of the box the
            string is displayed in. The string will be broken
            automatically to fit those boxes. If None, the string
            will not be broken.
            Note that this argument is incompatible with the fixed_size
            argument.
        """
        self.fixed_size = fixed_size
        self.box_size = box_size

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the string type from a rom.
        
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
        string : str
            The string in the rom.
        """
        string, _ = project.coder.hex_to_str(rom, offset)
        return string

    def to_assembly(self, string, project, context, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the pointer type.
        
        Parameters:
        -----------
        string : str
            The string to assemble.
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
            The assembly representation of the string.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        directives = project.config['string']['as']['directives']
        if self.fixed_size is not None:
            assembly = f'{directives["padded"]} {self.fixed_size} "{string}"'
        elif self.box_size is not None:
            width, height = self.box_size
            assembly = f'{directives["auto"]} {width} {height} "{string}"'
        else:
            assembly = f'{directives["std"]} "{string}"'
        return label_and_align(assembly, label, alignment, global_label), []
        
    
    def __call__(self, project, context, parents):
        """ Initializes a new string. 
        
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
        string : str
            Empty string.
        """
        return ''
    
    def size(self, string, project, context, parents):
        """ Returns the size of the string.

        Parameters:
        -----------
        string : str
            The string
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
        size : int
            The size of this type in bytes.
        """
        if self.fixed_size is not None:
            return self.fixed_size
        return len(project.coder.str_to_hex(string))

    def get_constants(self, string, project, context, parents):
        """ Returns a set of all constants that are used by this type and
        potential subtypes.
        
        Parameters:
        -----------
        string : str
            The string.
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
        return set()

# Define dict of lambdas to retrieve scalar types
scalar_from_data = {
    'u8' : (lambda rom, offset, project: struct.unpack_from('<B', rom, offset=offset)[0]),
    's8' : (lambda rom, offset, _: struct.unpack_from('<b', rom, offset=offset)[0]),
    'u16' : (lambda rom, offset, _: struct.unpack_from('<H', rom, offset=offset)[0]),
    's16' : (lambda rom, offset, _: struct.unpack_from('<h', rom, offset=offset)[0]),
    'u32' : (lambda rom, offset, _: struct.unpack_from('<I', rom, offset=offset)[0]),
    's32' : (lambda rom, offset, _: struct.unpack_from('<i', rom, offset=offset)[0]),
    'pointer' : (lambda rom, offset, project: (
        struct.unpack_from('<I', rom, offset=offset)[0] - project['rom']['offset'] if
        struct.unpack_from('<I', rom, offset=offset)[0] != 0 else None
    ))
}

# Define dict to export a scalar to assembly
scalar_to_assembly = {
    'u8' : (lambda value : f'.byte {value}'),
    's8' : (lambda value : f'.byte ({value} & 0xFF)'),
    'u16' : (lambda value : f'.hword {value}'),
    's16' : (lambda value : f'.hword ({value} & 0xFFFF)'),
    'u32' : (lambda value : f'.word {value}'),
    's32' : (lambda value : f'.word {value}'),
    'pointer' : (lambda value : f'.word {(value if value is not None else 0)}')
}

# Define the lenght of scalars
scalar_to_length = {
    'u8' : 1,
    's8' : 1,
    'u16' : 2,
    's16' : 2,
    'u32' : 4,
    's32' : 4,
    'pointer' : 4 
}

def associate_with_constant(value, proj, constant):
    """ Tries to associate a value form a constant table of the pymap project.
    
    Parameters:
    -----------
    value : int
        The value to associate with a constant
    proj : pymap.project.Project
        The pymap project to contain the constant tables.
    constant : str or None
            The constant table this type will be associated with.

    Returns:
    --------
    associated : int or str
        The association of the value or the original value if no association was possible.
    """
    if constant is not None:
        if constant in proj.constants:
            constants = proj.constants[constant]
            for key in constants:
                if constants[key] == value:
                    return key
            warn(f'No match for value {value} found in constant table {constant}')
        else:
            warn(f'Constant table {constant} not found in project.')
    return value

def label_and_align(assembly, label, alignment, global_label):
    """ Adds label and alignment to an assembly representation of a type.
    
    Parameters:
    -----------
    assembly : str
        The assembly representation of the type
    label : str or None
        The label of the type if requested
    alignment : int or None
        The alignment of the type if requested
    global_label : bool
        If the label generated will be exported globally.
        Only relevant if label is not None.
    
    Returns:
    --------
    assembly : str
        The aligned and labeled assembly representation of the type
    """
    blocks = []
    if alignment is not None:
        blocks.append(f'.align {alignment}')
    if label is not None:
        if global_label:
            blocks.append(f'.global {label}')
        blocks.append(f'{label}:')
    blocks.append(assembly)
    return '\n'.join(blocks)


# Initialize common types
u8 = ScalarType('u8')
u16 = ScalarType('u16')
u32 = ScalarType('u32')
s8 = ScalarType('s8')
s16 = ScalarType('s16')
s32 = ScalarType('s32')
pointer = ScalarType('pointer')
