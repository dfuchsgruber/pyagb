#!/usr/bin/env python3

import agb.string.compile
from functools import partial
import argparse
import pymap.project

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses an assembly or C(++) file.')
    parser.add_argument('input', help='Path to the input file.')
    parser.add_argument('project', help='Path to the project file.')
    parser.add_argument('-o', dest='output', help='Path to the output file.')
    parser.add_argument('--filetype', dest='file_type', help='Filetype, either c or asm. Default: infered', default=None)
    args = parser.parse_args()

    file_type = args.file_type
    if file_type is None:
        if args.input.endswith('.s') or args.input.endswith('.asm'):
            file_type = 'asm'
        elif args.input.endswith('.c') or args.input.endswith('.cpp'):
            file_type = 'c'
        else:
            raise RuntimeError(f'Unable to infer file type from path {file_type}')

    project = pymap.project.Project(args.project)

    if file_type in ('asm', 's'):
        agb.string.compile.preprocess_assembly(args.input, project, args.output)
    elif file_type in ('c', 'cpp'):
        agb.string.compile.preprocess_cpp(args.input, project, args.output)
    else:    
        raise RuntimeError(f'Unkown file type {file_type}')
