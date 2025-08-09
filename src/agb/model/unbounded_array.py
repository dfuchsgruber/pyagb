"""Variable length arrays, terminated by a sentiel value."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymap.project import Project

from agb.model.type import ModelContext, ModelParents, ModelValue, Type, label_and_align


class UnboundedArrayType(Type):
    """Variable length arrays, terminated by a sentiel value."""

    def __init__(self, datatype: str, sentinel: ModelValue):
        """Initializes the array type of variable length.

        Parameters:
        -----------
        datatype : str
            The datatype the pointer is pointing to.
        sentinel : object
            An instanciation of the datatype that serves as sentinel for the array.
        """
        self.datatype = datatype
        self.sentinel = sentinel

    def from_data(
        self,
        rom: bytearray,
        offset: int,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> ModelValue:
        """Retrieves the array from a rom.

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
        values: list[ModelValue] = []
        parents = list(parents) + [values]
        datatype = project.model[self.datatype]
        idx = 0
        while True:
            value = datatype.from_data(
                rom, offset, project, list(context) + [idx], parents
            )
            if value == self.sentinel:
                break
            values.append(value)
            offset += datatype.size(value, project, list(context) + [idx], parents)
            idx += 1
        return values

    def to_assembly(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
        label: str | None = None,
        alignment: int | None = None,
        global_label: bool = False,
    ) -> tuple[str, list[str]]:
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
        parents = list(parents) + [value]
        datatype = project.model[self.datatype]
        assert isinstance(value, list), f'Expected a list, got {value}'
        for i, value_i in enumerate(value + [self.sentinel]):
            block, additional = datatype.to_assembly(
                value_i, project, list(context) + [i], parents
            )
            if i == len(value):
                blocks.append(f'@ sentinel: {self.sentinel}')
            blocks.append(f'{block} @ {i}')
            additional_blocks += additional
        assembly = '\n'.join(blocks)
        return label_and_align(
            assembly, label, alignment, global_label
        ), additional_blocks

    def __call__(
        self, project: Project, context: ModelContext, parents: ModelParents
    ) -> ModelValue:
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
        return []

    def size(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> int:
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
        assert isinstance(value, list), f'Expected a list, got {value}'
        size = 0
        parents = list(parents) + [value]
        datatype = project.model[self.datatype]
        for i, value_i in enumerate(value + [self.sentinel]):
            # Sum each element individually (this is more clean...)
            size += datatype.size(value_i, project, list(context) + [i], parents)
        return size

    def get_constants(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> set[str]:
        """All constants (recursively) required to export this value (if any).

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
        constants: set[str] = set()
        parents = list(parents) + [value]
        datatype = project.model[self.datatype]
        # Only using the first element would be faster, but this approach
        # is more clean and versatile.
        assert isinstance(value, list), f'Expected a list, got {value}'
        for i, value_i in enumerate(value):
            constants.update(
                datatype.get_constants(value_i, project, list(context) + [i], parents)
            )
        return constants
