#!/usr/bin/env python3

import argparse
import json
import pymap.project
import pymap.compile

# Command line script to compile pymap structures. Those include
# headers (.pmh), footers (.pmf), tilesets (.pts) and also the 
# project file itself (.pmh).

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compiles pymap files.')
    generic_group = parser.add_argument_group('Compile generic file types')
    generic_group.add_argument('input', help='The file to compile.')
    project_group = parser.add_argument_group('Compile project files')
    parser.add_argument('project', help='The project file.')
    parser.add_argument('-o', dest='output', help='The output assembly file to create.')
    parser.add_argument('-p', '--project', help='Compile a project file.', action='store_true', dest='compile_project', default=False)
    project_group.add_argument('--headertable', help='Label for the map header table.', dest='header_table_label')
    project_group.add_argument('--footertable', help='Label for the map footer table.', dest='footer_table_label')
    args = parser.parse_args()

    # Load project
    project = pymap.project.Project(args.project)

    if args.compile_project:
        assembly = pymap.compile.project_to_assembly(project, args.header_table_label, args.footer_table_label)
    else:
        # Compile any datatype
        with open(args.input) as f:
            input = json.load(f)
        assembly = pymap.compile.datatype_to_assembly(input['data'], input['type'], input['label'], project)
        
    with open(args.output, 'w+') as f:
        f.write(assembly)
