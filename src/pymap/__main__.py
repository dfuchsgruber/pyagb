"""Entry points for the pymap command line interface."""

import argparse
import json
from pathlib import Path

import agb.string.compile

import pymap.compile
import pymap.project
from pymap.gui.main import gui
from pymap.constants import Constants


def pymap_gui_cli():
    """Entry point for the pymap gui application."""
    # os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1.5'  # Support for High DPI
    gui.main()


def pymap2s_cli():
    """Entry point for compiling pymap files."""
    parser = argparse.ArgumentParser(description='Compiles pymap files.')
    generic_group = parser.add_argument_group('Compile generic file types')
    generic_group.add_argument('input', help='The file to compile.')
    project_group = parser.add_argument_group('Compile project files')
    parser.add_argument('project', help='The project file.')
    parser.add_argument('-o', dest='output', help='The output assembly file to create.')
    parser.add_argument(
        '-p',
        '--project',
        help='Compile a project file.',
        action='store_true',
        dest='compile_project',
        default=False,
    )
    project_group.add_argument(
        '--headertable',
        help='Label for the map header table.',
        dest='header_table_label',
    )
    project_group.add_argument(
        '--footertable',
        help='Label for the map footer table.',
        dest='footer_table_label',
    )
    args = parser.parse_args()

    # Load project
    project = pymap.project.Project(args.project)

    if args.compile_project:
        assembly = pymap.compile.project_to_assembly(
            project, args.header_table_label, args.footer_table_label
        )
    else:
        # Compile any datatype
        with open(args.input, encoding=project.config['json']['encoding']) as f:
            input = json.load(f)
        assembly = pymap.compile.datatype_to_assembly(
            input['data'], input['type'], input['label'], project
        )

    with open(args.output, 'w+') as f:
        f.write(assembly)


def pymap_export_constants_cli():
    """Entry point for exporting constants."""
    parser = argparse.ArgumentParser(
        description='Exports constants into either assembly or C header files.'
    )
    parser.add_argument('input', help='Path to the input file.')
    parser.add_argument('output', help='Path to the output file.')
    parser.add_argument(
        '-l', '--label', dest='label', help='Label for the constants.', default=None
    )
    parser.add_argument(
        '-t',
        '--filetype',
        dest='file_type',
        help='Filetype,' ' either c or asm. Default: infered',
        default=None,
    )
    args = parser.parse_args()

    file_type = args.file_type
    if file_type is None:
        if args.input.endswith('.s') or args.input.endswith('.asm'):
            file_type = 'asm'
        elif args.input.endswith('.c') or args.input.endswith('.cpp'):
            file_type = 'c'
        else:
            raise RuntimeError(f'Unable to infer file type from path {file_type}')

    if args.label is None:
        label = Path(args.input).stem
    else:
        label = args.label

    constants = Constants({label: Path(args.input)})[label]

    # Format the values
    if file_type == 'as':
        macro = (
            '\n'.join([f'.equ {const}, {value}' for const, value in constants.items()])
            + '\n'
        )
    elif file_type == 'c':
        macro = '\n'.join(
            [
                f'#ifndef H_CONST_{label.upper()}',
                f'#define H_CONST_{label.upper()}',
                f'enum {label} ' + '{',
            ]
            + [f'{const} = {value},' for const, value in constants.items()]
            + ['};', '#endif']
        )
    else:
        raise RuntimeError(f'Unkown export type {type}')

    with open(args.output, 'w+') as f:
        f.write(macro)
        print(f'Exported constants {label} to {args.output}')


def bin2s_cli():
    """Entry point for the bin2s script."""
    parser = argparse.ArgumentParser(
        description='Converts a binary file to an ' 'assembly file.'
    )
    parser.add_argument('input', help='Path to the input file.')
    parser.add_argument('-o', dest='output', help='Path to the output file.')
    parser.add_argument('-s', dest='symbol', help='Symbol name.', default=None)
    args = parser.parse_args()

    symbol = args.symbol or Path(args.output).stem

    with open(args.input, 'rb') as f:
        bytes = bytearray(f.read())

    # Create assembly
    assembly = '.align 4\n.global {0}\n{0}:\n.byte {1}\n'.format(
        symbol, ', '.join(map(hex, bytes))
    )

    with open(args.output, 'w+') as f:
        f.write(assembly)


def pypreproc_cli():
    """Entry point for the pypreproc script."""
    parser = argparse.ArgumentParser(
        description='Preprocesses an assembly or ' 'C(++) file.'
    )
    parser.add_argument('input', help='Path to the input file.')
    parser.add_argument('project', help='Path to the project file.')
    parser.add_argument('-o', dest='output', help='Path to the output file.')
    parser.add_argument(
        '--filetype',
        dest='file_type',
        help='Filetype, either ' 'c or asm. Default: infered',
        default=None,
    )
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
