# Module that encapsulates compiling map headers, map footers, tilesets and pymap projects

import json
import agb.types
import pymap.model.header, pymap.model.footer, pymap.model.tileset

def get_preamble(constants_to_import, project):
    """ Creates the preamble for an assembly file, i.e.
    the include directives.

    Parameters:
    -----------
    constants : list of str
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

def header_to_assembly(header, label, project):
    """ Creates an assembly based on a map header file.
    
    Parameters:
    -----------
    header : dict 
        The map header to compile.
    label : str
        The label of the map header.
    project : pymap.project.Project
        The pymap project associated with the map.

    Returns:
    --------
    assembly : str
        The assembly of the map header.
    """
    header_type = project.model['header.header']
    assembly, additional_blocks = header_type.to_assembly(header, project, [], label=label, alignment=2, global_label=True)
    blocks = [get_preamble(project.config['pymap2s']['include']['header'], project), assembly] + additional_blocks + ['']
    return '\n\n'.join(blocks)

def footer_to_assembly(footer, label, project):
    """ Creates an assembly based on a map footer file.
    
    Parameters:
    -----------
    footer : dict 
        The map footer to compile.
    label : str
        The label of the footer.
    project : pymap.project.Project
        The pymap project associated with the map.

    Returns:
    --------
    assembly : str
        The assembly of the map footer.
    """
    footer_type = project.model['footer.footer']
    assembly, additional_blocks = footer_type.to_assembly(footer, project, [], label=label, alignment=2, global_label=True)
    blocks = [get_preamble(project.config['pymap2s']['include']['footer'], project), assembly] + additional_blocks + ['']
    return '\n\n'.join(blocks)

def tileset_to_assembly(tileset, label, project, is_primary):
    """ Creates an assembly based on a tileset file.
    
    Parameters:
    -----------
    tileset : dict 
        The tileset to compile.
    label : str
        The label of the tileset.
    project : pymap.project.Project
        The pymap project associated with the map.
    is_primary : bool
        If the tileset is a primary tileset.

    Returns:
    --------
    assembly : str
        The assembly of the tileset.
    """
    tileset_type = project.model['tileset.tileset_primary' if is_primary else 'tileset.tileset_secondary']
    assembly, additional_blocks = tileset_type.to_assembly(tileset, project, [], label=label, alignment=2, global_label=True)
    blocks = [get_preamble(project.config['pymap2s']['include']['tileset'], project), assembly] + additional_blocks + ['']
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
            num_maps = max(map(int, bank.keys())) + 1
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

        

