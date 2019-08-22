#!/usr/bin/env python3

""" This module is able to export the constants of a project """

from pymap import constants, project
import os, sys, getopt


def main(args):
    """ Exports the constants of a project as header / macro files """

    try:
        opts, args = getopt.getopt(args, 'ht:', ['help'])
    except getopt.GetoptError:
        sys.exit(2)
    
    type = None
    label = None
    # Parse opts
    for opt, arg in opts:
        if opt in ('-ht:l:', '--help'): 
            print('Usage: pymapconstex.py [opts] file.const output\nOptions:\n-t type\t\t either "c" or "as" (default=from filename)\n-l label\t\tName of the enumeration (default=from filename)')
            exit(0)
        elif opt in ('-t'):
            type = arg
        elif opt in ('-l'):
            label = arg
        else:
            raise RuntimeError(f'Unrecognized option {opt}')
    
    infile = args[0]
    outfile = args[1]
    if type is None:
        # Infer output file type
        if outfile.endswith('.c'):
            type = 'c'
        elif outfile.endswith('s') or outfile.endswith('as'):
            type = 'asm'
        else:
            raise RuntimeError(f'Could not infer output file type from {outfile}')
    if label is None:
        _, label = os.path.split(infile)
        label = label[:label.find('.')]
    
    # Parse and output the constants file
    with open(infile) as f:
        table = eval(f.read())

    # Output the constants in the desired format
    if table['type'] == 'enum':
        values = [(const, i + table.get('base', 0)) for i, const in enumerate(table['values'])]
    elif table['type'] == 'dict':
        values = [(value, table['values'][value]) for value in table['values']]
    else:
        raise RuntimeError(f'Unkown table type {table["type"]}!')
    # Format the values
    if type == 'as':
        macro = '\n'.join([f'.equ {const}, {value}' for const, value in values]) + '\n'
    elif type == 'c':
        macro = '\n'.join([f'#ifndef H_CONST_{label.upper()}', f'#define H_CONST_{label.upper()}', 
        f'enum {label} ' + '{'] + [f'{const} = {value},' for const, value in values] + ['};', '#endif'])
    else:
        raise RuntimeError(f'Unkown export type {type}')
    
    with open(outfile, 'w+') as f:
        f.write(macro)
        print(f'Exported constants {label} to {outfile}')
    


if __name__ == "__main__":
    main(sys.argv[1:])
