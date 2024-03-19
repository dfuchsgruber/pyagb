"""A bitfield type for scalar types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymap.project import Project

from agb.model.scalar_type import ScalarType, scalar_to_assembly
from agb.model.type import (
    ModelContext,
    ModelParents,
    ModelValue,
    associate_with_constant,
    label_and_align,
)


class BitfieldType(ScalarType):
    """Class for bitfield types."""

    def __init__(
        self,
        fmt: str,
        structure: list[tuple[str, str | None, int]],
        hidden_members: set[str] = set(),
    ):
        """Initializes the bitfield type.

        Parameters:
        -----------
        fmt : str
            A string {s/u}{bitlength} that encodes the scalar type that underlies
            the bitfield. Example: 'u8', 's32', ...
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
        hidden_members : set
            The members to hide when displaying the bitfield in a parameter tree.
        """
        super().__init__(fmt, constant=None)
        self.structure = structure
        self.hidden_members = hidden_members

    def from_data(
        self,
        rom: bytearray,
        offset: int,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> ModelValue:
        """Initializes the bitfield type from a rom.

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
        scalar_value = super().from_data(rom, offset, project, context, parents)
        assert isinstance(scalar_value, int), f'Expected an integer, got {scalar_value}'
        value: dict[str, ModelValue] = {}
        for member, constant, size in self.structure:
            mask = (1 << size) - 1
            member_value = associate_with_constant(
                (scalar_value >> bit_idx) & mask, project, constant
            )
            value[member] = member_value
            bit_idx += size
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
        assert isinstance(value, dict), f'Expected a dictionary, got {value}'
        shifted: list[str] = []
        bit_idx = 0
        for member, _, size in self.structure:
            value_i = value[member]
            mask = (1 << size) - 1
            shifted.append(f'(({value_i} & {mask}) << {bit_idx})')
            bit_idx += size
        assembly = scalar_to_assembly[self.fmt](' | '.join(shifted))
        return label_and_align(assembly, label, alignment, global_label), []

    def __call__(
        self, project: Project, context: ModelContext, parents: ModelParents
    ) -> ModelValue:
        """Initializes a new empty bitfield.

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
        return {member: 0 for member, _, _ in self.structure}

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
        return set(
            [constant for _, constant, _ in self.structure if constant is not None]
        )
