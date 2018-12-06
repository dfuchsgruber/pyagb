from agb.model.scalar_type import ScalarType, scalar_to_assembly
from agb.model.type import label_and_align, associate_with_constant

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
        value : dict
            The values at the given offset in rom associated with a constant string if possible.
        """
        bit_idx = 0
        scalar_value = super().from_data(rom, offset, proj, context, parents)
        value = {}
        for member, constant, size in self.structure:
            mask = (1 << size) - 1
            member_value = associate_with_constant((scalar_value >> bit_idx) & mask, proj, constant)
            value[member] = member_value
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
        shifted = []
        bit_idx = 0
        for member, _, size in self.structure:
            value = values[member]
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
        return {member : 0 for member, constant, size in self.structure}

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