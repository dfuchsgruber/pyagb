from agb.model.type import Type, associate_with_constant, label_and_align
import agb.string.agbstring
import agb.string.compile

class StringType(Type):
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
