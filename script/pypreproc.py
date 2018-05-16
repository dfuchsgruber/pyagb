#!/usr/bin/env python3

import os.path, sys, getopt
from pokestring import preprocasm, preprocc

if __name__ == "__main__":
     # Setup argparser
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:o:", ["help"])
    except getopt.GetoptError:
        sys.exit(2)

    filetype = None
    output = None

    # Parse opts
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("""Usage pypreproc.py [opts] input charmap [macro]

Arguments:
    input           :   Input file path
    charmap         :   Charmap file path
    [macro]         :   Macro which strings must be contained
                        in. Required for filetype 'c'

Options:
    -t {filetype}   :   Filetype ('c' or 'asm') [infered]
    -o {path}       :   Output file

Supported assembly directives:
    1. Translates a string into .byte section
        Example: .string "foo"
    2. Translates a normal string and pads it with zeros
        Example: .stringpad <length> "foo"
    3. Translates a string and automatically inserts
    linebreaks / linescrolls where needed. If no more
    lines fit in the textbox a scrolling is inserted
    as linebreak. Linebreaks are only performed at
    spaces (sequences mapping to 0x0).
        Example: .autostring <linewidth> <lines per box> "foo"
    
Supported c / cpp directives:
    MACRO("foo", "bar", ...)

    All strings that are given as macro parameters will be
    translated into an char array independent of any language.
    The macro itself therefore must resolve the language
    selection.

    

            """)
            exit(0)
        elif opt in ("-t"):
            filetype = arg
        elif opt in ("-o"):
            output = arg
        else:
            raise Exception("Unkown option {0}!".format(opt))
    
    if len(args) < 1:
        raise Exception("No input specified (see --help).")
    if len(args) < 2:
        raise Exception("No charmap file specified (see --help).")
    
    input = args[0]
    charmap = args[1]
    
    if not output:
        raise Exception("No output specified (see --help).")
    
    if not filetype:
        # Infer filetype
        _, extension = os.path.splitext(input)
        if extension in (".s", ".asm"):
            filetype = "asm"
        elif extension in (".c", ".cpp"):
            filetype = "c"
        else:
            raise Exception("Could not infere filetype from \
            extension {0}.".format(extension))
    
    if filetype == "asm":
        preprocasm.preprocess_assembly(input, charmap, output)
    else:
        if len(args) < 3:
            raise Exception("No macro specified (see --help).")
        macro = args[2]
        preprocc.preprocess_c(input, charmap, output, macro)