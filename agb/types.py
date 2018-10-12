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
            default : object
                The default initialization
        """
        self.structure = structure

    def from_data(self, rom, offset, project, context):
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

        Returns:
        --------
        structure : dict
            The initialized structure
        offset : int
            The offset after all scalar members of the struct have been processed
        """
        structure = {}
        for attribute, type, _ in self.structure:
            # Use the callback
            value, offset = type.from_data(rom, offset, project, context)
            structure[attribute] = value
        return structure, offset

    def to_assembly(self, structure, label=None, alignment=None):
        """ Returns an assembly representation of a structure.

        structure : dict
            The structure to convert to an assembly string.
        label : string or None
            The label to export (only if not None).
        alignment : int or None
            The alignment of the structure if required
            
        Returns:
        --------
        assembly : str
            The assembly representation.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        assemblies, additional_blocks = zip(*(datatype.to_assembly(getattr(self, attribute)) for attribute, datatype, default in self.structure))

        # Concatenate all additional blocks to a new list of 
        additional_blocks = sum(additional_blocks, [])
        
        return label_and_align('\n'.join(assemblies), label, alignment), additional_blocks

    def to_dict(self):
        """ Returns a dict representation of the structure.
        
        Returns:
        --------
        attributes : dict
            Mapping from attributes -> values.
         """
        return {attribute : getattr(self, attribute) for attribute, _, _ in self.structure}
    
    def from_dict(self, attributes):
        """ 
        Initializes the structure from a dictionary of attributes.
        Note that this dict does
        not initialize the structural information self.structure which
        instead has to be passed to the constructor. 

        Parameters:
        -----------
        attributes : dict
            The attributes and values for the event."""
        for key in attributes:
            setattr(self, key, attributes[key])

class ScalarType:
    """ Class to encapsulte scalar types. """

    def __init__(self, fmt, constant=None):
        """ Initializes a scalar type. 
        
        Parameters:
        -----------
        fmt : str
            A string {s/u}{bitlength} that encodes the scalar type.
            Example: 'u8', 's32', ...
        constant : str or None
            The constant table this type will be associated with.
        """
        self.fmt = fmt
        self.constant = constant
    
    
    def from_data(self, rom, offset, proj, context):
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
        
        Returns:
        --------
        value : int
            The value at the given offset in rom.
        offset : int
            The offeset of the next bytes after the value was processed
        """
        value, offset = scalar_from_data[self.fmt](rom, offset, proj)

        # Try to associate the value with a constant
        value = associate_with_constant(value, proj, self.constant)
        return value, offset

    @staticmethod
    def to_assembly(value, label=None, alignment=None):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        value : string or int
            The value to export
        label : string or None
            The label to export (only if not None).
        alignment : int or None
            The alignment of the structure if required
            
        Returns:
        --------
        assembly : str
            The assembly representation.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        label_and_align(scalar_to_assembly[self.fmt](value), label, align), []


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

    def from_data(self, rom, offset, proj, context):
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

    def to_assembly(self, values):
        """ Returns an assembly instruction line to export this scalar type.
        
        Parameters:
        -----------
        values : list of string or int
            The values to export
            
        Returns:
        --------
        assembly : str
            The assembly representation instruction line
        """
        if not len(values) == len(self.structure):
            raise RuntimeError(f'More values for bitfield provided {len(values)} than structure supports {len(self.structure)}')
        shifted = []
        bit_idx = 0
        for value, triplet in zip(values, self.structure):
            member, _, size = triplet # Unpack the member information
            mask = (1 << size) - 1
            shifted.append(f'(({value} << {bit_idx}) & mask)')
        return scalar_to_assembly[self.fmt](' | '.join(shifted))

class UnionType:
    """ Class for union type. """

    def __init__(self, subtypes, name_get):
        """ Initializes the union type.
        
        Parameters:
        -----------
        subtypes : dict
            Dict mapping from subtype names to the respective subtypes
            of the union. The keys are of type str, while the values
            are tuples of Type instances (ScalarType, Structure, ...) and
            default values. 
        name_get : function
            Function that returns the name of the subtype that is currently
            used for the union. The union type will be exported to assembly
            for this type and only values for this subtype will be retrieved
            considering the from_data function. Therefore values of other
            subtypes remain default initialized.
        """
        self.subtypes = subtypes
        self.name_get = name_get
    
    def from_data(self, rom, offset, project, context):
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
        
        Returns:
        --------
        values : dict
            Dict that maps from the subtype names to their value instanciation.
        offset : int
            The maximum offset after each subtype was consumed.
        """
        values = {}
        if self.name_get() not in self.subtypes:
            raise RuntimeError(f'Active subtype of union {self.name_get()} not part of union type.')
        for name in self.subtypes:
            subtype, default = self.subtypes[name]
            if name == self.name_get():
                value, offset_processed = subtype.from_data(rom, offset, project, context)
                values[name] = value
            else:
                values[name] = default

        return values, offset_processed

    def to_assembly(self, values):
        """ Creates an assembly representation of the union type.
        
        Parameters:
        -----------
        values : dict
            Dict that maps from the subtype names to their value instanciaton.

        Returns:
        --------
        assembly : str
            The assembly representation of the specific subtype.
        """
        active_subtype_name = self.name_get()
        active_subtype, _ = self.subtypes[active_subtype_name]
        return active_subtype.to_assembly(values[active_subtype_name])
        


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

def label_and_align(assembly, label, alignment):
    """ Adds label and alignment to an assembly representation of a type.
    
    Parameters:
    -----------
    assembly : str
        The assembly representation of the type
    label : str or None
        The label of the type if requested
    alignment : int or None
        The alignment of the type if requested
    
    Returns:
    --------
    assembly : str
        The aligned and labeled assembly representation of the type
    """
    blocks = []
    if alignment is not None:
        blocks.append(f'.align {alignment}')
    if label is not None:
        blocks.append(f'.global {label}')
        blocks.append(f'{label}:')
    blocks.append(assembly)
    return '\n'.join(blocks)