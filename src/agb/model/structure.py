"""This module contains the structure type."""

from pymap.project import Project

from agb.model.type import ModelContext, ModelParents, ModelValue, Type, label_and_align


class Structure(Type):
    """Superclass to model any kind of structure."""

    def __init__(self, structure: list[tuple[str, str, int]],
                 hidden_members: set[str]=set()):
        """Initialize the members with priorized members.

        Parameters:
        -----------
        structure : list of tuples (member, type)
            member : str
                The name of the member
            type : str
                The name of the type
            priority : int
                Members with lower priority will get processed first

        hidden_members : set
            The members to hide when displaying the structure in a parameter tree.
        """
        self.structure = structure
        self.hidden_members = hidden_members

    def from_data(self, rom: bytearray, offset: int, project: Project,
                  context: ModelContext, parents: ModelParents) -> ModelValue:
        """Retrieves the structure from a rom.

        Parameters:
        -----------
        rom : bytearray
            The rom to initialize the structure from
        offset : int
            The offset to initialize the structure from
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list of str
            The context in which the data got initialized
        parents : list
            The parent values of this value. The last
            element is the direct parent. The parents are
            possibly not fully initialized as the values
            are explored depth-first.

        Returns:
        --------
        structure : dict
            The initialized structure
        """
        structure: dict[str, ModelValue] = {}
        parents = parents + [structure]

        priorities = sorted(list(set([x[2] for x in self.structure])))
        for priority in priorities: # Export according to priorities
            for attribute, datatype_name, datatype_priority in self.structure:
                datatype = project.model[datatype_name]
                if datatype_priority == priority:
                    # Export the member in this iteration
                    value = datatype.from_data(rom, offset, project,
                                               context + [attribute], parents)
                elif datatype_priority > priority and attribute not in structure:
                    # Initilize the datatype with an empty stub in order to retrieve
                    # its size
                    value = datatype(project, context + [attribute], parents)
                else:
                    value = structure[attribute]
                structure[attribute] = value
                # Initialize empty before exporting
                offset += datatype.size(value, project, context + [attribute], parents)
        return structure

    def __call__(self, project: Project, context: ModelContext,
                 parents: ModelParents) -> ModelValue:
        """Initializes a new structure with default values.

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
        structure : dict
            New structure with default values.
        """
        structure: dict[str, ModelValue] = {}
        parents = parents + [structure]
        # Initialize priorized members first
        for attribute, datatype_name, _ in sorted(self.structure, key=lambda x: x[2]):
            structure[attribute] = project.model[datatype_name](project,
                                                                context + [attribute],
                                                                parents)
        return structure


    def to_assembly(self, value: ModelValue, project: Project, context: ModelContext,
                    parents: ModelParents, label: str | None=None,
                    alignment: int | None=None,
                    global_label: bool=False) -> tuple[str, list[str]]:
        """Returns an assembly representation of a structure.

        structure : dict
            The structure to convert to an assembly string.
        project : pymap.project.Project
            The project to e.g. fetch constant from
        context : list
            Context from parent elements.
        parents : list
            The parent values of this value. The last
            element is the direct parent.
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
        assemblies: list[str] = []
        additional_blocks: list[str] = []
        assert isinstance(value, dict), f"Expected a dict, got {value}"
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            assembly_datatype, additional_blocks_datatype = datatype.to_assembly(
                value[attribute], project, context + [attribute], parents + [value])
            assemblies.append(assembly_datatype)
            additional_blocks += additional_blocks_datatype

        return label_and_align('\n'.join(assemblies), label, alignment, global_label), \
            additional_blocks

    def size(self, value: ModelValue, project: Project, context: ModelContext,
             parents: ModelParents) -> int:
        """Returns the size of a specific structure instanze in bytes.

        Parameters:
        -----------
        structure : dict
            The structure of which the size is desired.
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
        size = 0
        parents = parents + [value]
        assert isinstance(value, dict), f"Expected a dict, got {value}"
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            size += datatype.size(value[attribute], project, context + [attribute],
                                  parents)
        return size

    def get_constants(self, value: ModelValue, project: Project, context: ModelContext,
                      parents: ModelParents) -> set[str]:
        """All constants (recursively) required by this structure.

        Parameters:
        -----------
        structure : dict
            The structure of which the size is desired.
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
        parents = parents + [value]
        assert isinstance(value, dict), f"Expected a dict, got {value}"
        for attribute, datatype_name, _ in self.structure:
            datatype = project.model[datatype_name]
            constants.update(datatype.get_constants(value[attribute],
                                                    project,
                                                    context + [attribute],
                                                    parents))
        return constants

