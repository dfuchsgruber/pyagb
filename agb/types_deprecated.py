from itertools import chain
from abc import ABC

class Type(ABC):
    """ Abstract superclass for types. """

    def __init__(self):
        pass

    def from_data(self, rom, offset, parent, proj):
        """ Initializes an instance of this type from binary data

        Parameters:
        -----------
        rom : agb.agbrom.Agbrom
            The rom that contains the binary data
        offset : int
            The offset of the binary data in the rom.
        parent : str or None
            The name of the parent structure if existent
        proj : pymap.project.Project
            The pymap project from which all structures can
            be accessed. Note that the parent structure
            is expected to be fully initialized.

        Returns:
        --------
        offset : int
            The offset after the instance was processed
        substructures : list of triplets
            All substructures that have to be exported recursively:
            label : str
                The label the substructure will be assigned with
            offset : int
                The offest of the data the substructure will be exported from
            datatype : Type
                The datatype of the substructure that will be exported.
        """
        raise NotImplementedError

    def to_assembly(self):
        """ Returns an assembly representation of an instance of this type
        
        Parameters:
        -----------
        instance : object
            The instance to retrun an assembly represenation of
        
        Returns:
        --------
        representation : str
            The assembly representation of the string.
        """
        raise NotImplementedError
    
    def serialize(self):
        """ Provides a json serializable representation of the type.
        
        Returns:
        --------
        serialized : list, str, dict, float, int, ...
            Serialized representation of type instance
        """
        raise NotImplementedError
    
    def deserialize(self, serialization):
        """ Initializes this instance of a type from its json serialization.
        
        Parameters:
        -----------
        serialization : list, str, dict, float, int, ...
            Serialized representation of the type instance
        """
        raise NotImplementedError

class ScalarType(Type):
    """ Superclass for scalar types. """

    def __init__(self):
        self.value = 0

    def serialize(self):
        return self.value
    


types = {
    # Unsigned char (u8)
    'u8' : (
        (lambda rom, offset, parent, proj: (rom.u8[offset], offset + 1, [])),
        (lambda value : f'.byte {value}')
    ),
    # Signed char (s8)
    's8' : (
        (lambda rom, offset, parent, proj: (rom.s8[offset], offset + 1, [])),
        (lambda value : f'.byte ({value} & 0xFF)')
    ),
    # Unsigned short (u16)
    'u16' : (
        (lambda rom, offset, parent, proj: (rom.u16[offset], offset + 2, [])),
        (lambda value : f'.hword {value}')
    ),
    # Signed short (u16)
    's16' : (
        (lambda rom, offset, parent, proj: (rom.s16[offset], offset + 2, [])),
        (lambda value : f'.hword ({value} 0xFFFF)')
    ),
    # Unsigned int32 (u32)
    'u32' : (
        (lambda rom, offset, parent, proj: (rom.u32[offset], offset + 4, [])),
        (lambda value : f'.word {value}')
    ),
    # Signed int32 char (int)
    'int' : (
        (lambda rom, offset, parent, proj: (rom.int[offset], offset + 4, [])),
        (lambda value : f'.hword ({value} 0xFFFFFFFF)')
    )
}

# Functors for composite datatypes (pointers, arrays, etc.)

def make_array(datatype, size_getter):
    """ Creates an array datatype based on another datatype.
    
    Parameters:
    -----------
    datatype : type-interface (see doc above)
        The datatype that the array consists of.
    size_getter : function
        Function that provides the size of the array.

            Parameters:
            -----------
            rom : Agbrom
                Rom where the array will be extracted from.
            offset : int
                Offset of the first element.
            parent : string
                Label of the parent structure (has all
                its field initialized when size_getter
                is called).
            proj : pymap.Project
                The pymap project from which all structures, 
                i.e. the parent can be accessed.

            Returns:
            --------
            size : int
                The number of elements in the array.

    Returns:
    --------
    array_type : type-interface (see doc above)
        The composite array datatype.
    """
    def from_data(rom, offset, parent, proj):
        # Helper method to retrieve the array
        size = size_getter(rom, offset, parent, proj)
        values = []
        substructures = []
        for _ in range(size):
            # Export this structure
            value, offset, substructure = proj.types[datatype][0](rom, offset, parent, proj)
            values.append(value)
            substructures.append(substructure)
        return values, offset, substructures

    def to_assembly(value):
        # Helper method to export to assembly
        return '\n'.join(map(datatype[1], values))

    return from_data, to_assembly

def make_pointer(datatype, label_generator):
    """ Creates a pointer datatype that points to an another datatype.
    
    Parameters:
    -----------
    datatype : type-interface
        The datatype that the pointer points towards.
    label_generator : function
        Function that provides the label for the type
        the pointer points towards.

            Parameters:
            -----------
            rom : Agbrom
                Rom where the array will be extracted from.
            offset : int
                Offset of the first element.
            parent : string
                Label of the parent structure (has all
                its field initialized when size_getter
                is called).
            proj : pymap.Project
                The pymap project from which all structures, 
                i.e. the parent can be accessed.

            Returns:
            --------
            label : str
                The label of value / structure the pointer
                points towards.

    Returns:
    -------
    pointer_type : type-interface (see doc above)
        The composite pointer datatype.
    """
    def from_data(rom, offset, parent, proj):
        # Helper method to retrieve the pointer
        label = label_generator(rom, offset, parent, proj)
        # Trigger a new structure exploration
        target_offset = rom.pointer[offset]
        return label, offset + 4, [(label, target_offset, datatype)]

    def to_assembly(value):
        return f'.word {value}'
    
    return from_data, to_assembly


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
        for attribute, _, default in self.structure:
            setattr(self, attribute, default)

    def from_data(self, rom, offset, parent, proj):
        substructures = []
        for attribute, type, _ in self.structure:
            # Use the callback
            value, offset, substructure = types[type][0](rom, offset, parent, proj)
            substructures += substructure
            setattr(self, attribute, value)
        return self, offset, substructures

    def to_assembly(self):
        """ Returns an assembly representation of the structure.
        
        Returns:
        --------
        assembly : string
            The assembly representation of the instance.
        """
        return '\n'.join(
            (datatype[1](getattr(self, attribute)) for attribute, datatype, default in self.structure)
        )

    def to_dict(self):
        """ Returns a dict representation of the structure.
        
        Returns:
        --------
        attribute : dict
            Mapping from attributes -> values. Note that this dict does
            not initialize the structural information self.structure which
            instead has to be passed to the constructor.
         """
        return self.__dict__
    
    def from_dict(self, attributes):
        """ Initializes the structure from a dictionary of attributes. 
        Parameters:
        -----------
        attributes : dict
            The attributes and values for the event."""
        for key in attributes:
            setattr(self, key, attributes[key])


def make_structure(structure):
    """ Creates a structure type.
    
    Parameters:
    -----------
    structure : list of triplets (member, type, default)
        member : str
            The name of the member
        type : str
            The name of the type
        default : object
            The default initialization

    Returns:
    -------
    struct_type : type-interface (see doc above)
        The composite struct datatype.
    """
    return (
        (lambda rom, offset, parent, proj: Structure(structure).from_data(rom, offset, parent, proj)),
        (lambda struct: struct.to_assembly())
    )
        
