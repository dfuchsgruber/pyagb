"""Module to model pointers to other data types."""

from typing import Callable

from pymap.project import Project

from agb.model.scalar_type import ScalarType
from agb.model.type import ModelContext, ModelParents, ModelValue


class PointerType(ScalarType):
    """Class to models pointers."""

    def __init__(self, datatype: str,
                 label_get: Callable[
                     [Project, ModelContext, ModelParents],
                     tuple[str, int, bool]]):
        """Initializes the pointer to another datatype.

        Parameters:
        -----------
        datatype : str
            The datatype the pointer is pointing to.
        label_get : function
            Function that creates the label for the structure.

            Parameters:
            -----------
            project : pymap.project.Project
                The map project
            context : list
                Context from parent elements.
            parents : list
                The parent values of this value. The last
                element is the direct parent. The parents are
                possibly not fully initialized as the values
                are explored depth-first.

        Returns:
            --------
            label : str or None
                The label of the data the pointer points to.
            alignment : int or None
                The alignment of the data that the pointer points to.
            global_label : bool
                If the data's label will be exported globally.
        """
        super().__init__('pointer', constant=None)
        self.datatype = datatype
        self.label_get = label_get

    def from_data(self, rom: bytearray, offset: int, project: Project,
                  context: ModelContext, parents: ModelParents) -> ModelValue:
        """Retrieves the pointer type from a rom.

        Parameters:
        -----------
        rom : bytearray
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
        """
        data_offset =  super().from_data(rom, offset, project, context, parents)
        if data_offset is None:
            # Nullpointer
            return None
        assert isinstance(data_offset, int), f'Expected an integer, got {data_offset}'
        # Retrieve the data
        datatype = project.model[self.datatype]
        return datatype.from_data(rom, data_offset, project, context, parents)

    def to_assembly(self, value: ModelValue, project: Project, context: ModelContext,
                    parents: ModelParents, label: str | None=None,
                    alignment: int | None=None,
                    global_label: bool=False) -> tuple[str, list[str]]:
        """Creates an assembly representation of the pointer type.

        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
            The assembly representation of the pointer.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        if value is None:
            # Nullpointer
            return super().to_assembly(value, project, context, parents, label=label,
                                       alignment=alignment, global_label=global_label)

        data_label, data_alignment, data_global_label = self.label_get(project, context,
                                                                       parents)
        assembly, additional_blocks = super().to_assembly(data_label, project, context,
                                                          parents, label=label,
                                                          alignment=alignment,
                                                          global_label=global_label)
        # Create assembly for the datatype that the pointer refers to
        datatype = project.model[self.datatype]
        data_assembly, data_additional_blocks = datatype.to_assembly(value, project,
                                                                     context, parents,
                                                                     label=data_label,
                                                                     alignment=data_alignment,
                                                                     global_label=data_global_label)
        # The data assembly is an additional block as well
        additional_blocks.append(data_assembly)
        additional_blocks += data_additional_blocks
        return assembly, additional_blocks

    def __call__(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> ModelValue:
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
        data : object
            Default intialized object.
        """
        datatype = project.model[self.datatype]
        return datatype(project, context, parents)


    def get_constants(self, value: ModelValue, project: Project, context: ModelContext,
                      parents: ModelParents) -> set[str]:
        """All constants (recursively) required to export this value (if any).

        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
        datatype = project.model[self.datatype]
        if value is None:
            return set()
        return datatype.get_constants(value, project, context, parents)


class DynamicLabelPointer(PointerType):
    """Class to model pointers that have no fixed label but can change their label."""

    @staticmethod
    def label_get(project: Project, context: ModelContext, parents: ModelParents) -> tuple[str, int, bool]:
        """Creates a label for the pointer.

        Parameters:
        -----------
        project : pymap.project.Project
            The map project
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.

        Returns:
        --------
        label : str
            The label of the data the pointer points to.
        alignment : int
            The alignment of the data that the pointer points to.
        global_label : bool
            If the data's label will be exported globally.
        """
        return '', 1, False

    def __init__(self, datatype: str, data_alignment: int | None,
                 data_label_global: bool, prefix: str='loc_'):
        """Initializes the dynamic pointer type.

        Parameters:
        -----------
        datatype : str
            The datatype the pointer is pointing to.
        data_alignment : int or None
            The alignment of the data that the pointer points to.
        data_label_global : bool
            If the data's label will be exported globally.
        prefix : str
            The prefix of the label that is generated from an offset. E.g., for a prefix 'loc_',
            if the pointer refers to 0x100, the label will be 'loc_100'
        """
        super().__init__(datatype, self.label_get)
        self.data_alignment = data_alignment
        self.data_label_global = data_label_global
        self.prefix = prefix

    def from_data(self, rom: bytearray, offset: int, project: Project,
                  context: ModelContext, parents: ModelParents) -> ModelValue:
        """Retrieves the pointer type from a rom.

        Parameters:
        -----------
        rom : bytearray
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
        label : str
            The label of the data.
        data : object
            The data the pointer is pointing to.
        """
        data = super().from_data(rom, offset, project, context, parents)
        return f'{self.prefix}{hex(offset)[2:]}', data

    def to_assembly(self, value: ModelValue, project: Project, context: ModelContext,
                    parents: ModelParents, label: str | None = None,
                    alignment: int | None=None,
                    global_label: bool=False) -> tuple[str, list[str]]:
        """Creates an assembly representation of the pointer type.

        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
            The assembly representation of the pointer.
        additional_blocks : list of str
            Additional assembly blocks that resulted in the recursive
            compiliation of this type.
        """
        assert isinstance(value, tuple), f'Expected a tuple, got {value}'
        data_label, data = value
        if data is None:
            # Nullpointer
            return super().to_assembly(data, project, context, parents, label=label,
                                       alignment=alignment, global_label=global_label)

        assembly, additional_blocks = super().to_assembly(data_label, project, context,
                                                          parents, label=label,
                                                          alignment=alignment,
                                                          global_label=global_label)
        # Create assembly for the datatype that the pointer refers to
        datatype = project.model[self.datatype]
        data_assembly, data_additional_blocks = datatype.to_assembly(data, project,
                                                                     context,parents,
                                                                     label=data_label,
                                                                     alignment=self.data_alignment,
                                                                     global_label=self.data_label_global)
        # The data assembly is an additional block as well
        additional_blocks.append(data_assembly)
        additional_blocks += data_additional_blocks
        return assembly, additional_blocks

    def __call__(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> ModelValue:
        """Initializes a new pointer.

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
        label : str
            The default label prefix_{'default'}
        data : object
            Default intialized object.
        """
        return f'{self.prefix}default', super()(project, context, parents)

    def get_constants(self, value: ModelValue, project: Project, context: ModelContext,
                      parents: ModelParents) -> set[str]:
        """All constants (recursively) required to export this value (if any).

        Parameters:
        -----------
        data : object
            The data the pointer is pointing to.
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
        assert isinstance(value, tuple), f'Expected a tuple, got {value}'
        _, data = value
        return super().get_constants(data, project, context, parents)
