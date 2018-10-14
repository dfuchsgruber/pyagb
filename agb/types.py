# Import and export functions for scalar types and non-scalar types

from warnings import warn

class Structure:
    """ Superclass to model any kind of structure """
    def __init__(self, structure):
        """ Initializes the structure class from a given structure.
        
        Parameters:
        -----------
        structure : list of tuples (member, type, default)
            member : str
                The name of the member
            type : str
                The name of the type
        """
        self.structure = structure

    def from_data(self, rom, offset, project, context, parents):
        """ Initializes all members as scalars from the rom. 
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
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
        for attribute, datatype in self.structure:
            value = datatype.from_data(rom, offset, project, context + [attribute], parents + [structure])
            offset += len(datatype)
            structure[attribute] = value
        return structure

    def to_assembly(self, structure, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly representation of a structure.

        structure : dict
            The structure to convert to an assembly string.
        label : string or None
            The label to export (only if not None).
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
        for attribute, datatype in self.structure:
            assembly_datatype, additional_blocks_datatype = datatype.to_assembly(structure[attribute], parents + [structure])
            assemblies.append(assembly_datatype)
            additional_blocks += additional_blocks_datatype
        
        return label_and_align('\n'.join(assemblies), label, alignment, global_label), additional_blocks

    def __call__(self, parents):
        """ Initializes a new structure with default values. 
        
        Parameters:
        -----------
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
        for attribute, datatype in self.structure:
            structure[attribute] = datatype(parents + [structure])
        return structure

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
        rom : agb.agbrom.Agbrom
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
        offset : int
            The offeset of the next bytes after the value was processed
        """
        value, _ = scalar_from_data[self.fmt](rom, offset, proj)

        # Try to associate the value with a constant
        value = associate_with_constant(value, proj, self.constant)
        return value, offset

    def to_assembly(self, value, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        value : string or int
            The value to export
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

    def __call__(self, parents):
        """ Returns a new empty value (0). 
        
        Parameters:
        -----------
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


class BitfieldType:
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
        self.fmt = fmt
        self.structure = structure

    def from_data(self, rom, offset, proj, context, parents):
        """ Retrieves the constant type from a rom and tries to associate the constant value.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
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
        offset : int
            The offeset of the next bytes after the value was processed
        """
        bit_idx = 0
        scalar_value, offset = scalar_from_data[self.fmt](rom, offset, proj)
        value = []
        for member, constant, size in self.structure:
            mask = (1 << size) - 1
            member_value = associate_with_constant((scalar_value >> bit_idx) & mask, proj, constant)
            value.append(member_value)
            bit_idx += size
        return value, offset

    def to_assembly(self, values, parents, label=None, alignment=None, global_label=False):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        values : list of string or int
            The values to export
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
            shifted.append(f'(({value} << {bit_idx}) & {mask})')
            bit_idx += size
        assembly = scalar_to_assembly[self.fmt](' | '.join(shifted))
        return label_and_align(assembly, label, alignment, global_label), []

    def __call__(self, parents):
        """ Initializes a new empty bitfield.
        
        Parameters:
        -----------
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

class UnionType:
    """ Class for union type. """

    def __init__(self, subtypes, name_get):
        """ Initializes the union type.
        
        Parameters:
        -----------
        subtypes : dict from str -> Type
            Dict mapping from subtype names to the respective subtypes
            of the union. Keys are the subtype names str and map to
            types such as ScalarType, BitfieldType, etc.
        name_get : function
            Function that returns the name of the subtype that is currently
            used for the union. The union type will be exported to assembly
            for this type and only values for this subtype will be retrieved
            considering the from_data function. Therefore values of other
            subtypes remain default initialized.
            
            Parameters:
            -----------
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
        rom : agb.agbrom.Agbrom
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
        offset : int
            The maximum offset after each subtype was consumed.
        """
        values = {}
        if self.name_get(parents) not in self.subtypes:
            raise RuntimeError(f'Active subtype of union {self.name_get()} not part of union type.')
        for name in self.subtypes:
            if name == self.name_get(parents):
                value, offset_processed = self.subtypes[name].from_data(rom, offset, project, context + [name], parents + [values])
                values[name] = value
            else:
                values[name] = self.subtypes[name](parents)

        return values, offset_processed

    def to_assembly(self, values, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the union type.
        
        Parameters:
        -----------
        values : dict
            Dict that maps from the subtype names to their value instanciaton.
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
        active_subtype_name = self.name_get(parents)
        active_subtype = self.subtypes[active_subtype_name]
        return active_subtype.to_assembly(values[active_subtype_name], parents + [values], label=label, alignment=alignment, global_label=global_label)

    def __call__(self, parents):
        """ Initializes a new empty union type. 
        
        Parameters:
        -----------
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
        return {name : self.subtypes[name](parents) for name in self.subtypes}
        
class ArrayType:
    """ Type for arrays. """

    def __init__(self, datatype, size_get):
        """ Initializes the array type.
        
        Parameters:
        -----------
        datatype : type
            The underlying datatype
        size_get : function
            Function that returns the number of elements in the array. This
            function can access all parents of the array type, but not
            all parent's elements will be initialized, as the exporting
            of types is depth-first.
            Parameters:
            -----------
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
        rom : agb.agbrom.Agbrom
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
        offset : int
            The offset after all elements in the array were consumed.
        """
        num_elements = self.size_get(parents)
        values = []
        for i in range(num_elements):
            value, offset = self.datatype.from_data(rom, offset, project, context + [str(i)], parents + [values])
            values.append(value)
        return values, offset
    
    
    def to_assembly(self, values, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the union type.
        
        Parameters:
        -----------
        values : list
            The values of the array
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
        for i in range(self.size_get(parents)):
            block, additional = self.datatype.to_assembly(values[i], parents + [values])
            blocks.append(block)
            additional_blocks += additional
        assembly = '\n'.join(blocks)
        return label_and_align(assembly, label, alignment, global_label), additional_blocks

    def __call__(self, parents):
        """ Initializes a new array. 
        
        Parameters:
        -----------
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
        for _ in range(self.size_get(parents)):
            values.append(self.datatype(parents + [values]))
        return values

class PointerType:
    """ Class to models pointers. """

    def __init__(self, datatype, label_get):
        """ Initializes the pointer to another datatype.
        
        Parameters:
        -----------
        datatype : Type
            The datatype the pointer is pointing to.
        label_get : function
            Function that creates the label for the structure.
            
            Returns:
            --------
            label : str or None
                The label of the data the pointer points to.
            alignment : int or None
                The alignment of the data that the pointer points to.
            global_label : bool
                If the data's label will be exported globally.
            """
        self.datatype = datatype
        self.label_get = label_get

    def from_data(self, rom, offset, project, context, parents):
        """ Retrieves the pointer type from a rom.
        
        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
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
        offset : int
            The offset after the pointer was consumed.
        """
        data_offset, offset = pointer.from_data(rom, offset, project, context, parents)
        # Retrieve the data
        data, _ = self.datatype.from_data(rom, data_offset, project, context, parents)
        return data, offset
    
    def to_assembly(self, data, parents, label=None, alignment=None, global_label=None):
        """ Creates an assembly representation of the pointer type.
        
        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
        data_label, data_alignment, data_global_label = self.label_get()
        assembly, additional_blocks = pointer.to_assembly(data_label, parents, label=label, alignment=alignment, global_label=global_label)
        # Create assembly for the datatype that the pointer refers to
        data_assembly, data_additional_blocks = self.datatype.to_assembly(data, parents, label=data_label, alignment=data_alignment, global_label=data_global_label)
        # The data assembly is an additional block as well
        additional_blocks.append(data_assembly)
        additional_blocks += data_additional_blocks
        return assembly, data_additional_blocks

    def __call__(self, parents):
        """ Initializes a new array. 
        
        Parameters:
        -----------
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
        return self.datatype()


# Define dict of lambdas to retrieve scalar types
scalar_from_data = {
    'u8' : (lambda rom, offset, _: (rom.u8[offset], offset + 1)),
    's8' : (lambda rom, offset, _: (rom.s8[offset], offset + 1)),
    'u16' : (lambda rom, offset, _: (rom.u16[offset], offset + 2)),
    's16' : (lambda rom, offset, _: (rom.s16[offset], offset + 2)),
    'u32' : (lambda rom, offset, _: (rom.u32[offset], offset + 4)),
    'int' : (lambda rom, offset, _: (rom.int[offset], offset + 4)),
    'pointer' : (lambda rom, offset, _: (rom.pointer[offset], offset + 4))
}

# Define dict to export a scalar to assembly
scalar_to_assembly = {
    'u8' : (lambda value : f'.byte {value}'),
    's8' : (lambda value : f'.byte ({value} & 0xFF)'),
    'u16' : (lambda value : f'.hword {value}'),
    's16' : (lambda value : f'.hword ({value} & 0xFFFF)'),
    'u32' : (lambda value : f'.word {value}'),
    'int' : (lambda value : f'.word {value}'),
    'pointer' : (lambda value : f'.word {value}')
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
