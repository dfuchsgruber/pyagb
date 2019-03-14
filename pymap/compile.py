# Module that encapsulates compiling map headers, map footers, tilesets and pymap projects

import json
import agb.types
import pymap.model.header, pymap.model.footer, pymap.model.tileset


def get_preamble(constants_to_import, project):
    """ Creates the preamble for an assembly file, i.e.
    the include directives.

    Parameters:
    -----------
    constants : set of str
        The name of the constant tables to include in the assembly
    project : pymap.project.Project
        The pymap project.

    Returns:
    --------
    preamble : str
        The include preamble
    """
    config = project.config['pymap2s']['include']
    return '\n'.join(config['directive'].replace('{constant}', constant) for constant in constants_to_import)


def datatype_to_assembly(data, datatype, label, project):
    """ Creates an assembly based on any generic datatype.

    Parameters:
    -----------
    data : object 
        An instance of the datatype.
    datatype : str
        The datatype that is to be compiled.
    label : str
        The label of the map header.
    project : pymap.project.Project
        The pymap project associated with the map.

    Returns:
    --------
    assembly : str
        The assembly of the datatype.
    """
    datatype = project.model[datatype]
    constants = datatype.get_constants(data, project, ['get_constants'], [])
    assembly, additional_blocks = datatype.to_assembly(data, project, ['to_assembly'], [], label=label, alignment=2, global_label=True)
    blocks = [get_preamble(constants, project), assembly] + additional_blocks + ['']
    return '\n\n'.join(blocks)


def project_to_assembly(project, header_table_label, footer_table_label):
    """ Creates an assembly based on a project file.
    
    project : pymap.project.Project
        The pymap project to assemble.
    header_table_label : str
        The label for the map table.
    footer_table_label : str
        The label for the map footer table.

    Returns:
    --------
    assembly : str
        The assembly of the project.
    """
    blocks = []

    # Create the map table
    num_banks = max(map(int, project.headers.keys())) + 1
    map_table_assembly = []
    for i in map(str, range(num_banks)): # @Todo: maybe enable user defined translation based on e.g. constants
        if i in project.headers:
            map_table_assembly.append(f'.word bank_{i}')
            # Create a new bank
            bank = project.headers[i]
            map_idxs = set(map(int, bank.keys()))
            map_idxs.add(-1) # To support empty banks
            num_maps = max(map_idxs) + 1
            bank_assembly = []
            for j in map(str, range(num_maps)): # @Todo: maybe enable user defined translation based on e.g. constants
                if j in bank:
                    label, _, _ = bank[j]
                    bank_assembly.append(f'.word {label}')
                else:
                    bank_assembly.append('.word 0')
            blocks.append(agb.types.label_and_align('\n'.join(bank_assembly), f'bank_{i}', 2, False))
        else:
            map_table_assembly.append('.word 0')
    blocks = [agb.types.label_and_align('\n'.join(map_table_assembly), header_table_label, 2, True)] + blocks

    # Create the footer table
    footer_idx_to_label = {project.footers[label][0] : label for label in project.footers}
    footer_table_assembly = [
        f'.word {footer_idx_to_label[i]}' if i in footer_idx_to_label else '.word 0' 
        for i in range(1, max(map(int, footer_idx_to_label.keys())) + 1) # Footers start at 1 (idx 0 is reserved)
    ]
    blocks.append(agb.types.label_and_align('\n'.join(footer_table_assembly), footer_table_label, 2, True))

    return '\n\n'.join(blocks + [''])

        

