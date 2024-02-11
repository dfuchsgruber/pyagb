"""Datatype for arrays."""

from typing import Callable

from pymap.project import Project

from agb.model.type import (
    ModelContext,
    ModelParents,
    ModelValue,
    ScalarModelValue,
    Type,
    label_and_align,
)


class ArrayType(Type):
    """Type for arrays."""

    def __init__(self, datatype: str, fixed_size: bool):
        """Initializes the array type.

        Parameters:
        -----------
        datatype : str
            The underlying datatype
        fixed_size : bool
            If the size of the array is fixed.
        """
        self.datatype = datatype
        self.fixed_size = fixed_size

    def size_get(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> int:
        """Retrieves the size of the array.

        Parameters:
        -----------
        project : pymap.project.Project
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
        size : int
            The size of the array.
        """
        raise NotImplementedError()

    def from_data(self, rom: bytearray, offset: int, project: Project,
                  context: ModelContext, parents: ModelParents) -> list[ModelValue]:
        """Retrieves the type value from a rom.

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
        values: list[ModelValue] = []
        datatype: Type = project.model[self.datatype]
        for i in range(num_elements):
            value = datatype.from_data(rom, offset, project, context + [i],
                                       parents + [values])
            values.append(value)
            offset += datatype.size(value, project, context + [i], parents + [values])
        return values


    def to_assembly(self, value: ModelValue, project: Project, # type: ignore
                    context: ModelContext, parents: ModelParents,
                    label: str | None=None, alignment: int | None=None,
                    global_label: bool = False) -> tuple[str, list[str]]:
        """Creates an assembly representation of the union type.

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
        blocks: list[str] = []
        additional_blocks: list[str] = []
        datatype = project.model[self.datatype]
        assert isinstance(value, list)
        for i in range(self.size_get(project, context, parents)):
            block, additional = datatype.to_assembly(value[i], project, context + [i],
                                                     parents + [value])
            blocks.append(block)
            additional_blocks += additional
        assembly = '\n'.join(blocks)
        return label_and_align(assembly, label, alignment, global_label), \
            additional_blocks

    def __call__(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> list[ModelValue]:
        """Initializes a new array.

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
        values: list[ModelValue] = []
        datatype = project.model[self.datatype]
        for i in range(self.size_get(project, context, parents)):
            values.append(datatype(project, context + [i], parents + [values]))
        return values

    def size(self, value: ModelValue, project: Project, context: ModelContext,
             parents: ModelParents) -> int:
        """Returns the size of a specific structure instanze in bytes.

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
        parents = parents + [value]
        datatype = project.model[self.datatype]
        assert isinstance(value, list)
        for i in range(num_elements):
            # Sum each element individually (this is more clean...)
            size += datatype.size(value[i], project, context + [i], parents)
        return size

    def get_constants(self, value: ModelValue, project: Project, context: ModelContext,
                      parents: ModelParents) -> set[str]:
        """All constants (recursively) used by that type.

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
        constants: set[str] = set()
        assert isinstance(value, list)
        parents = parents + [value]
        datatype = project.model[self.datatype]
        # Only using the first element would be faster, but this approach
        # is more clean and versatile.
        for i in range(num_elements):
            constants.update(datatype.get_constants(value[i], project, context + [i], parents))
        return constants

class VariableSizeArrayType(ArrayType):
    """Type for variable size arrays."""

    def __init__(self, datatype: str,
                 size_path: tuple[int, list[str | int]],
                 size_cast: Callable[[ScalarModelValue, Project], int]=
                 lambda value, project: int(value)): # type: ignore
        """Initializes the array type.

        Parameters:
        -----------
        datatype : str
            The underlying datatype
        size_path : tuple
            Path to the size member in the parents. Each element of the list
            The tuple consists of n_parents : int and location : list.
            If n_parents is equal to i, then the i-th parent of the
            array is selected. This parents is then accessed successively
            by all elements in the sequence in location. Example:
            2, ['foo', 'bar'] describes grandparent['foo']['bar']
        size_cast : function
            A function that casts a size string to an integer. Signature:

            Parameters:
            -----------
            value : str
                The value to cast.
            proj : pymap.project.Project
                The pymap project to access e.g. constants.

        Returns:
            --------
            size : int
                The size of the array as integer.
        """
        super().__init__(datatype, fixed_size=False)
        self.size_path = size_path
        self.size_cast = size_cast

    def size_get(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> int:
        """Retrieves the size of the array.

        Parameters:
        -----------
        project : pymap.project.Project
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
        size : int
            The size of the array.
        """
        n_parents, location = self.size_path
        if n_parents <= 0:
            raise RuntimeError
        root = parents[-n_parents]
        for member in location:
            root = root[member] # type: ignore
        return self.size_cast(root, project) # type: ignore


class FixedSizeArrayType(ArrayType):
    """Type for fixed size arrays."""

    def __init__(self, datatype: str,
                 size_get: Callable[[Project, ModelContext], int]):
        """Initializes the array type with fixed size.

        Parameters:
        -----------
        datatype : str
            The underlying datatype
        size_get : function
            Function that retrieves the fixed size of the array. Signature:

            Parameters:
            -----------
            project : pymap.project.Project
                The pymap project to access e.g. constants.
            context : list of str
                The context in which the data got initialized.

        Returns:
            --------
            size : int
                The size of the array.
        """
        super().__init__(datatype, fixed_size=True)
        self.size_get = lambda project, context, parents: size_get(project, context)
