"""Scalar model type."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pymap.project import Project

from agb.model.type import (
    ModelContext,
    ModelParents,
    ModelValue,
    ScalarModelValue,
    Type,
    associate_with_constant,
    label_and_align,
)


class ScalarType(Type):
    """Class to model scalar types."""

    def __init__(
        self, fmt: str, constant: str | None = None, default: ScalarModelValue = 0
    ):
        """Initailizes the scalar type.

        Parameters:
        -----------
        fmt : str
            The format string for the scalar type. Either (u|s)(8|16|32) or 'pointer'.
        constant : str or None
            The constant table associated with the scalar type.
        default : str or int
            The default value.
        """
        self.fmt = fmt
        self.constant = constant
        self.default = default

    def from_data(
        self,
        rom: bytearray,
        offset: int,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> ModelValue:
        """Retrieves the scalar type from a rom.

        Parameters:
        -----------
        rom : bytearray
            The rom to retrieve the data from.
        offset : int
            The offset to retrieve the data from.
        project : pymap.project.Project
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
        value = scalar_from_data[self.fmt](rom, offset, project)

        # Try to associate the value with a constant
        value = associate_with_constant(value, project, self.constant)
        return value

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
        """Returns an assembly instruction line to export this scalar type.

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
        assert isinstance(value, ScalarModelValue)
        return label_and_align(
            scalar_to_assembly[self.fmt](value), label, alignment, global_label
        ), []

    def __call__(
        self, project: Project, context: ModelContext, parents: ModelParents
    ) -> ModelValue:
        """Returns a new empty value (0).

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
        Default value.
        """
        return self.default

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

    def get_constants(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> set[str]:
        """Returns (recursively) all constants that are used by this type.

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


def pointer_from_data(rom: bytearray, offset: int, project: Project) -> int | None:
    """Retrieves a pointer from a rom.

    Args:
        rom (bytearray): the rom to retrieve the data from.
        offset (int): the offset to retrieve the data from.
        project (Project): the project to access e.g. constants.

    Returns:
        int | None: the value at the given offset in rom.
    """
    offset = struct.unpack_from('<I', rom, offset=offset)[0]
    if offset > 0:
        offset -= project.config['rom']['offset']  # type: ignore
        return offset  # type: ignore
    else:
        return None


# Define dict of lambdas to retrieve scalar types
scalar_from_data: dict[str, Callable[[bytearray, int, Project], ScalarModelValue]] = {
    'u8': (
        lambda rom, offset, project: struct.unpack_from('<B', rom, offset=offset)[0]
    ),
    's8': (lambda rom, offset, _: struct.unpack_from('<b', rom, offset=offset)[0]),
    'u16': (lambda rom, offset, _: struct.unpack_from('<H', rom, offset=offset)[0]),
    's16': (lambda rom, offset, _: struct.unpack_from('<h', rom, offset=offset)[0]),
    'u32': (lambda rom, offset, _: struct.unpack_from('<I', rom, offset=offset)[0]),
    's32': (lambda rom, offset, _: struct.unpack_from('<i', rom, offset=offset)[0]),
    'pointer': pointer_from_data,
}

# Define dict to export a scalar to assembly
scalar_to_assembly: dict[str, Callable[[ScalarModelValue], str]] = {
    'u8': (lambda value: f'.byte {value}'),
    's8': (lambda value: f'.byte ({value} & 0xFF)'),
    'u16': (lambda value: f'.hword {value}'),
    's16': (lambda value: f'.hword ({value} & 0xFFFF)'),
    'u32': (lambda value: f'.word {value}'),
    's32': (lambda value: f'.word {value}'),
    'pointer': (lambda value: f'.word {(value if value is not None else 0)}'),
}

# Define the lenght of scalars
scalar_to_length = {
    'u8': 1,
    's8': 1,
    'u16': 2,
    's16': 2,
    'u32': 4,
    's32': 4,
    'pointer': 4,
}
