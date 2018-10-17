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
    return '\n'.join(config['directive'] for constant in constants_to_import)

def mapheader_to_assembly(map_header_path, assembly_path, project):
    """ Creates an assembly based on a map header file.
    
    Parameters:
    -----------
    map_header_path : str 
        Path to the map header file.
    assembly_path : str
        Path to the assembly file to create.
    project : pymap.project.Project
        The pymap project associated with the map.
    """
    # Load the map header file
    with open(map_header_path) as f:
        header = json.load(f)
    
    if header['type'] != 'header':
        raise RuntimeError(f'Expected a map header json file but got a {header["type"]!}')
    
    label = header['label']
    header_data = header['data']

    assembly, additional_blocks = pymap.model.header.header_type.to_assembly(header_data,
    [], label=label, alignment=2, global_label=True)
    blocks = [get_preamble(project.config['pymap2s']['include']['header'], project), assembly] + additional_blocks
    
    with open(assembly_path, 'w+') as f:
        f.write('\n\n'.join(blocks))    
