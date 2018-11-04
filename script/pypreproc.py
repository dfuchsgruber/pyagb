#!/usr/bin/env python3

import os.path, sys, getopt
from pokestring import preprocasm, preprocc

import agb.string.compile
from functools import partial
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses an assembly or C(++) file.')
    parser.add_argument('input', help='Path to the input file.')
    parser.add_argument('charmap', help='Path to the character map.')
    parser.add_argument('-o', dest='output', help='Path to the output file.')
    parser.add_argument('--filetype', dest='file_type', help='Filetype, either c or asm. Default: infered', default=None)
    parser.add_argument('--tail', dest='tail', help='Comma seperated list of integers representing a sequence terminating a string. Example: \'0x0,0xFF\'. Default: \'0xFF\'', default='0xFF')
    parser.add_argument('--macro', dest='macro', help='C(++) macro to enclose strings in. Example usage of MACRO: \'MACRO("...", "...")\'. Default: PSTRING ', default='PSTRING')
    args = parser.parse_args()

    file_type = args.file_type
    if file_type is None:
        if args.input.endswith('.s') or args.input.endswith('.asm'):
            file_type = 'asm'
        elif args.input.endswith('.c') or args.input.endswith('.cpp'):
            file_type = 'c'
        else:
            raise RuntimeError(f'Unable to infer file type from path {file_type}')
    tail = list(map(partial(int, base=0), args.tail.split(',')))

    if file_type in ('asm', 's'):
        agb.string.compile.preprocess_assembly(args.input, args.charmap, args.output, tail=tail)
    elif file_type in ('c', 'cpp'):
        agb.string.compile.preprocess_cpp(args.input, args.charmap, args.output, args.macro, tail)
    else:    
        raise RuntimeError(f'Unkown file type {file_type}')
