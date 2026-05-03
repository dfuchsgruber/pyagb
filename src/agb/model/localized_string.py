"""Type for strings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .string import StringType

if TYPE_CHECKING:
    from pymap.project import Project

from agb.model.type import ModelContext, ModelParents, ModelValue, label_and_align


class LocalizedStringType(StringType):
    """Type class for localized strings."""

    def __init__(
        self,
        fixed_size: int | None = None,
        box_size: tuple[int, int] | None = None,
        default_language: str = 'LANG_GER',
    ):
        """Initializes a localized string type.

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
        super().__init__(fixed_size, box_size)
        self.default_language = default_language

    def from_data(
        self,
        rom: bytearray,
        offset: int,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> ModelValue:
        """Retrieves the string type from a rom.

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
        return {
            self.default_language: super().from_data(
                rom, offset, project, context, parents
            )
        }

    def to_assembly(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelValue,
        label: str | None = None,
        alignment: int | None = None,
        global_label: bool = False,
    ) -> tuple[str, list[str]]:
        """Creates an assembly representation of the pointer type.

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
        assert isinstance(value, dict), f'Expected a dict, got {value}'
        assembly = ''
        additional_blocks = []
        template = project.config['string']['as']['localization_template']
        for language, text in value.items():
            assembly_text, blocks = super().to_assembly(
                text, project, context, parents, None, None, False
            )
            additional_blocks += blocks
            language_assembly = template.format(
                language=language, assembly=assembly_text
            )
            assembly += language_assembly + '\n'

        return label_and_align(
            assembly, label, alignment, global_label
        ), additional_blocks

    def __call__(
        self, project: Project, context: ModelContext, parents: ModelParents
    ) -> ModelValue:
        """Initializes a new string.

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
        return {self.default_language: ''}

    def size(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> int:
        """Returns the size of the string.

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
        assert isinstance(value, dict), f'Expected a dict, got {value}'
        value = value[self.default_language]
        return super().size(value, project, context, parents)

    def get_constants(
        self,
        value: ModelValue,
        project: Project,
        context: ModelContext,
        parents: ModelParents,
    ) -> set[str]:
        """All constants of this type (recursively) that are used by this type.

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
        assert isinstance(value, dict), f'Expected a dict, got {value}'
        return super().get_constants(
            value[self.default_language], project, context, parents
        )
